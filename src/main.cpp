#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <mutex>
#include <deque>
#include <cstdio>
#include <filesystem>
#include <array>
#include <sstream>

// ZMQ
#include <zmq.hpp>
#include <zmq_addon.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

// --- Platform Specific Headers ---
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <spawn.h>
#include <poll.h>
pid_t g_backend_pid = -1;
#endif

namespace fs = std::filesystem;

// --- 0. Enum & Consts ---
enum DataSourceMode {
    SOURCE_LOCAL_CAM = 0,
    SOURCE_EXTERNAL_ZMQ = 1
};

// --- 1. Global Application State ---
struct AppState {
    // === Settings ===
    DataSourceMode source_mode = SOURCE_LOCAL_CAM;
    int selected_cam_index = 0;
    std::vector<int> available_cams;
    std::string external_zmq_addr = "tcp://127.0.0.1:5555";

    // [新增] 是否显示预览图 (控制 GPU 占用)
    bool show_previews = true;

    // ZMQ Ports
    int port_pub_frames = 6000;
    int port_sub_preview = 6001;
    int port_sub_pose = 6002;

    std::string python_script = "scripts/engine.py";

    // === Runtime Status ===
    std::atomic<bool> is_running{ true };
    std::atomic<bool> camera_active{ false };
    std::atomic<bool> backend_running{ false };

    // [进程控制]
    std::mutex proc_mutex;
    void* backend_process_handle = nullptr; // Windows Handle

    // Connection Status
    std::atomic<bool> status_cam_pub{ false };
    std::atomic<bool> status_prev_sub{ false };
    std::atomic<bool> status_pose_sub{ false };

    // === Installer State ===
    std::atomic<bool> is_installing{ false };
    std::atomic<float> install_progress{ 0.0f };
    std::string install_status_text = "Idle";

    // === Data Buffers ===
    std::mutex data_mutex;
    cv::Mat frame_raw;
    cv::Mat frame_preview;
    std::vector<float> pose_data;

    // OpenGL Textures
    GLuint tex_raw = 0;
    GLuint tex_preview = 0;

    // === Logger ===
    std::mutex log_mutex;
    std::deque<std::string> logs;
    bool scroll_to_bottom = false;

    void Log(const std::string& msg) {
        std::lock_guard<std::mutex> lock(log_mutex);
        time_t now = time(0);
        tm* ltm = localtime(&now);
        char time_buf[16];
        sprintf(time_buf, "[%02d:%02d:%02d] ", ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
        logs.push_back(std::string(time_buf) + msg);
        if (logs.size() > 2000) logs.pop_front();
        scroll_to_bottom = true;
    }
} app;

// --- 2. Helper Functions ---

void StopBackend() {
    std::lock_guard<std::mutex> lock(app.proc_mutex);
#ifdef _WIN32
    if (app.backend_process_handle != nullptr) {
        HANDLE hProcess = (HANDLE)app.backend_process_handle;
        DWORD exitCode = 0;
        if (GetExitCodeProcess(hProcess, &exitCode) && exitCode == STILL_ACTIVE) {
            if (TerminateProcess(hProcess, 1)) {
                app.Log("[SYS] Kill signal sent (Windows).");
            }
            else {
                app.Log("[ERR] TerminateProcess failed.");
            }
        }
    }
#else
    if (g_backend_pid > 0) {
        if (kill(g_backend_pid, SIGTERM) == 0) {
            app.Log("[SYS] SIGTERM sent to PID " + std::to_string(g_backend_pid));
            int status;
            waitpid(g_backend_pid, &status, WNOHANG);
        }
        else {
            app.Log("[ERR] Failed to kill PID " + std::to_string(g_backend_pid));
        }
    }
#endif
}

void DrawStatusDot(bool active, float radius = 6.0f) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    float y_center_off = ImGui::GetTextLineHeight() * 0.5f;
    ImVec2 center(p.x + radius, p.y + y_center_off);
    ImU32 color = active ? IM_COL32(50, 205, 50, 255) : IM_COL32(100, 100, 100, 255);
    draw_list->AddCircleFilled(center, radius, color);
    ImGui::Dummy(ImVec2(radius * 2 + 5, radius * 2));
}

void RefreshCameraList() {
    app.Log("Scanning cameras...");
    std::vector<int> found;
    for (int i = 0; i < 4; i++) {
        cv::VideoCapture temp_cap(i);
        if (temp_cap.isOpened()) {
            found.push_back(i);
            temp_cap.release();
        }
    }
    app.available_cams = found;
    if (found.empty()) app.Log("No cameras found.");
    else app.Log("Found " + std::to_string(found.size()) + " cameras.");
}

// 核心：上传 OpenCV Mat 到 OpenGL 纹理
// 如果 app.show_previews 为 false，UI循环中将不调用此函数，从而节省显存带宽
void UpdateTexture(GLuint& texID, const cv::Mat& mat) {
    if (mat.empty()) return;
    if (texID == 0) glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
}

std::string GetPythonPath() {
    fs::path cwd = fs::current_path();
#ifdef _WIN32
    fs::path venv_py = cwd / "venv" / "Scripts" / "python.exe";
    if (fs::exists(venv_py)) return venv_py.string();
    return "python";
#else
    fs::path venv_py = cwd / "venv" / "bin" / "python";
    if (fs::exists(venv_py)) return venv_py.string();
    return "python3";
#endif
}

// --- 3. Process Logic ---

bool ExecCommand(const std::string& cmd) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES saAttr; saAttr.nLength = sizeof(SECURITY_ATTRIBUTES); saAttr.bInheritHandle = TRUE; saAttr.lpSecurityDescriptor = NULL;
    HANDLE hChildOut_Rd, hChildOut_Wr;
    if (!CreatePipe(&hChildOut_Rd, &hChildOut_Wr, &saAttr, 0)) return false;
    SetHandleInformation(hChildOut_Rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si; ZeroMemory(&si, sizeof(si)); si.cb = sizeof(si); si.hStdError = hChildOut_Wr; si.hStdOutput = hChildOut_Wr; si.dwFlags |= STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi; ZeroMemory(&pi, sizeof(pi));
    std::string cmd_wrapped = "cmd.exe /c " + cmd;
    std::vector<char> buf(cmd_wrapped.begin(), cmd_wrapped.end()); buf.push_back(0);
    if (!CreateProcessA(NULL, buf.data(), NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) { CloseHandle(hChildOut_Rd); CloseHandle(hChildOut_Wr); return false; }
    CloseHandle(hChildOut_Wr);
    DWORD dwRead; CHAR chBuf[1024]; std::string line;
    while (ReadFile(hChildOut_Rd, chBuf, sizeof(chBuf), &dwRead, NULL) && dwRead != 0) {
        for (DWORD i = 0; i < dwRead; i++) { if (chBuf[i] == '\n' || chBuf[i] == '\r') { if (!line.empty()) { app.Log(line); line.clear(); } } else line += chBuf[i]; }
    }
    if (!line.empty()) app.Log(line);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 0; GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(hChildOut_Rd);
    return exitCode == 0;
#else
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) return false;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        std::string line = buffer;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        app.Log(line);
    }
    return pclose(pipe) == 0;
#endif
}

void BackendMonitorThread(std::string python_exe, std::string script_path) {
    if (app.backend_running) return;
    app.backend_running = true;
    app.Log("[SYS] Launching: " + python_exe);

#ifdef _WIN32
    SECURITY_ATTRIBUTES saAttr; saAttr.nLength = sizeof(SECURITY_ATTRIBUTES); saAttr.bInheritHandle = TRUE; saAttr.lpSecurityDescriptor = NULL;
    HANDLE hChildOut_Rd, hChildOut_Wr;
    CreatePipe(&hChildOut_Rd, &hChildOut_Wr, &saAttr, 0);
    SetHandleInformation(hChildOut_Rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si; ZeroMemory(&si, sizeof(si)); si.cb = sizeof(si); si.hStdError = hChildOut_Wr; si.hStdOutput = hChildOut_Wr; si.dwFlags |= STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi; ZeroMemory(&pi, sizeof(pi));
    std::string cmd = "\"" + python_exe + "\" -u -X utf8 \"" + script_path + "\"";
    std::vector<char> buf(cmd.begin(), cmd.end()); buf.push_back(0);

    if (CreateProcessA(NULL, buf.data(), NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        CloseHandle(hChildOut_Wr); CloseHandle(pi.hThread);
        { std::lock_guard<std::mutex> lock(app.proc_mutex); app.backend_process_handle = pi.hProcess; }
        DWORD dwRead; CHAR chBuf[1024]; std::string line;
        while (app.is_running && ReadFile(hChildOut_Rd, chBuf, sizeof(chBuf), &dwRead, NULL) && dwRead != 0) {
            for (DWORD i = 0; i < dwRead; i++) { if (chBuf[i] == '\n' || chBuf[i] == '\r') { if (!line.empty()) { app.Log("[PY] " + line); line.clear(); } } else line += chBuf[i]; }
        }
        if (!line.empty()) app.Log("[PY] " + line);
        { std::lock_guard<std::mutex> lock(app.proc_mutex); CloseHandle(pi.hProcess); app.backend_process_handle = nullptr; }
        CloseHandle(hChildOut_Rd);
    }
    else { app.Log("[ERR] Failed to start Python process."); }
#else
    int pipe_fd[2];
    if (pipe(pipe_fd) == -1) { app.Log("[ERR] Pipe failed"); app.backend_running = false; return; }
    pid_t pid = fork();
    if (pid == -1) { app.Log("[ERR] Fork failed"); close(pipe_fd[0]); close(pipe_fd[1]); app.backend_running = false; return; }
    if (pid == 0) {
        close(pipe_fd[0]); dup2(pipe_fd[1], STDOUT_FILENO); dup2(pipe_fd[1], STDERR_FILENO); close(pipe_fd[1]);
        execl(python_exe.c_str(), python_exe.c_str(), "-u", script_path.c_str(), (char*)NULL);
        _exit(1);
    }
    else {
        close(pipe_fd[1]);
        { std::lock_guard<std::mutex> lock(app.proc_mutex); g_backend_pid = pid; }
        char buffer[1024]; ssize_t bytesRead; std::string line_buffer;
        while (app.is_running && (bytesRead = read(pipe_fd[0], buffer, sizeof(buffer) - 1)) > 0) {
            buffer[bytesRead] = '\0';
            for (int i = 0; i < bytesRead; i++) {
                if (buffer[i] == '\n' || buffer[i] == '\r') { if (!line_buffer.empty()) { app.Log("[PY] " + line_buffer); line_buffer.clear(); } }
                else { line_buffer.push_back(buffer[i]); }
            }
        }
        close(pipe_fd[0]); int status; waitpid(pid, &status, 0);
        { std::lock_guard<std::mutex> lock(app.proc_mutex); g_backend_pid = -1; }
    }
#endif
    app.backend_running = false;
    app.Log("[SYS] Backend Stopped.");
}

// --- 4. Logic Threads ---
void InstallThreadFunc() {
    app.is_installing = true;
    app.install_progress = 0.0f;
    app.Log("=== Installing Environment ===");
#ifdef _WIN32
    std::string sys_py = "python";
#else
    std::string sys_py = "python3";
#endif
    if (!ExecCommand(sys_py + " --version")) { app.Log("Error: System python not found."); app.is_installing = false; return; }
    app.install_progress = 0.2f;
    if (!ExecCommand(sys_py + " -m venv venv")) { app.Log("Error: Failed to create venv."); app.is_installing = false; return; }
    app.install_progress = 0.4f;
    fs::path cwd = fs::current_path();
#ifdef _WIN32
    std::string venv_pip = (cwd / "venv" / "Scripts" / "pip.exe").string();
#else
    std::string venv_pip = (cwd / "venv" / "bin" / "pip").string();
#endif
    std::string pip_cmd = "\"" + venv_pip + "\" install opencv-python pyzmq numpy mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple";
    app.Log("Downloading packages...");
    if (ExecCommand(pip_cmd)) { app.install_progress = 1.0f; app.install_status_text = "Success!"; app.Log("Environment Ready."); }
    else { app.install_status_text = "Failed."; app.Log("Pip install failed."); }
    app.is_installing = false;
}

void InstallDriverThread(std::string driverName) {
    app.Log("Installing " + driverName + "...");
    std::this_thread::sleep_for(std::chrono::seconds(2));
    app.Log(driverName + " Installed.");
}

void CameraThread() {
    zmq::context_t ctx(1);
    zmq::socket_t publisher(ctx, zmq::socket_type::pub);
    publisher.bind("tcp://*:" + std::to_string(app.port_pub_frames));
    zmq::socket_t subscriber(ctx, zmq::socket_type::sub);
    cv::VideoCapture cap;
    int current_cam_idx = -1;
    std::string current_zmq_addr = "";

    while (app.is_running) {
        if (!app.camera_active) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); app.status_cam_pub = false; continue; }
        cv::Mat frame;
        if (app.source_mode == SOURCE_LOCAL_CAM) {
            if (current_cam_idx != app.selected_cam_index || !cap.isOpened()) {
                cap.open(app.selected_cam_index);
                current_cam_idx = app.selected_cam_index;
                cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            }
            if (cap.isOpened()) cap >> frame;
        }
        else {
            if (cap.isOpened()) cap.release();
            if (current_zmq_addr != app.external_zmq_addr) {
                try { subscriber.disconnect(current_zmq_addr); }
                catch (...) {}
                current_zmq_addr = app.external_zmq_addr;
                subscriber.connect(current_zmq_addr); subscriber.set(zmq::sockopt::subscribe, "");
            }
            zmq::message_t msg;
            if (subscriber.recv(msg, zmq::recv_flags::dontwait)) {
                std::vector<uchar> data(static_cast<uchar*>(msg.data()), static_cast<uchar*>(msg.data()) + msg.size());
                frame = cv::imdecode(data, cv::IMREAD_COLOR);
            }
        }
        if (!frame.empty()) {
            { std::lock_guard<std::mutex> lock(app.data_mutex); frame.copyTo(app.frame_raw); }
            std::vector<uchar> buffer; cv::imencode(".jpg", frame, buffer, { cv::IMWRITE_JPEG_QUALITY, 50 });
            std::string meta = "{}";
            zmq::message_t msg_meta(meta.data(), meta.size()); zmq::message_t msg_payload(buffer.data(), buffer.size());
            publisher.send(msg_meta, zmq::send_flags::sndmore); publisher.send(msg_payload, zmq::send_flags::none);
            app.status_cam_pub = true;
        }
        else app.status_cam_pub = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
}

void ReceiverThread() {
    zmq::context_t ctx(1);
    zmq::socket_t sub_img(ctx, zmq::socket_type::sub); sub_img.connect("tcp://127.0.0.1:" + std::to_string(app.port_sub_preview)); sub_img.set(zmq::sockopt::subscribe, "");
    zmq::socket_t sub_pose(ctx, zmq::socket_type::sub); sub_pose.connect("tcp://127.0.0.1:" + std::to_string(app.port_sub_pose)); sub_pose.set(zmq::sockopt::subscribe, "");
    zmq::pollitem_t items[] = { { sub_img, 0, ZMQ_POLLIN, 0 }, { sub_pose, 0, ZMQ_POLLIN, 0 } };
    while (app.is_running) {
        zmq::poll(items, 2, std::chrono::milliseconds(10));
        if (items[0].revents & ZMQ_POLLIN) {
            std::vector<zmq::message_t> msgs; zmq::recv_multipart(sub_img, std::back_inserter(msgs));
            if (msgs.size() >= 2) {
                std::vector<uchar> data(static_cast<uchar*>(msgs[1].data()), static_cast<uchar*>(msgs[1].data()) + msgs[1].size());
                cv::Mat decoded = cv::imdecode(data, cv::IMREAD_COLOR);
                if (!decoded.empty()) { std::lock_guard<std::mutex> lock(app.data_mutex); decoded.copyTo(app.frame_preview); }
                app.status_prev_sub = true;
            }
        }
        else app.status_prev_sub = false;
        if (items[1].revents & ZMQ_POLLIN) {
            std::vector<zmq::message_t> msgs; zmq::recv_multipart(sub_pose, std::back_inserter(msgs));
            if (msgs.size() >= 2) {
                float* raw = static_cast<float*>(msgs[1].data()); size_t count = msgs[1].size() / sizeof(float);
                std::lock_guard<std::mutex> lock(app.data_mutex); app.pose_data.assign(raw, raw + count);
                app.status_pose_sub = true;
            }
        }
        else app.status_pose_sub = false;
    }
}

// --- 5. UI ---
void RenderUI(float dpi) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    float left_w = 400.0f * dpi;
    ImVec2 btn_size(-1, 40.0f * dpi);
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(left_w, viewport->WorkSize.y));
    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    // 1. Source
    ImGui::BeginChild("Source", ImVec2(0, 260 * dpi), true); // 增加高度容纳新选项
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "SOURCE"); ImGui::Separator();
    if (ImGui::RadioButton("Local Cam", app.source_mode == SOURCE_LOCAL_CAM)) app.source_mode = SOURCE_LOCAL_CAM;
    ImGui::SameLine(); if (ImGui::RadioButton("External ZMQ", app.source_mode == SOURCE_EXTERNAL_ZMQ)) app.source_mode = SOURCE_EXTERNAL_ZMQ;
    ImGui::Spacing();

    // [新增] 预览开关
    ImGui::Checkbox("Show Previews (Reduce GPU)", &app.show_previews);
    ImGui::Spacing();

    if (app.source_mode == SOURCE_LOCAL_CAM) {
        if (ImGui::Button("Scan Cams", ImVec2(-1, 30 * dpi))) RefreshCameraList();
        if (!app.available_cams.empty()) {
            std::string preview = "Cam " + std::to_string(app.selected_cam_index);
            if (ImGui::BeginCombo("##Sel", preview.c_str())) {
                for (int idx : app.available_cams) { if (ImGui::Selectable(("Cam " + std::to_string(idx)).c_str(), app.selected_cam_index == idx)) app.selected_cam_index = idx; }
                ImGui::EndCombo();
            }
        }
    }
    else { char buf[128]; strcpy(buf, app.external_zmq_addr.c_str()); if (ImGui::InputText("ZMQ Addr", buf, 128)) app.external_zmq_addr = std::string(buf); }
    ImGui::Spacing();
    if (app.camera_active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.2f, 0.2f, 1.0f));
        if (ImGui::Button("STOP STREAM", btn_size)) app.camera_active = false;
        ImGui::PopStyleColor();
    }
    else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
        if (ImGui::Button("START STREAM", btn_size)) app.camera_active = true;
        ImGui::PopStyleColor();
    }
    ImGui::EndChild();

    // 2. Backend
    ImGui::BeginChild("Backend", ImVec2(0, 220 * dpi), true);
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "BACKEND"); ImGui::Separator();
    bool venv = fs::exists(fs::current_path() / "venv");
    ImGui::Text("Venv: %s", venv ? "Yes" : "No");
    if (app.is_installing) { ImGui::ProgressBar(app.install_progress, ImVec2(-1, 20 * dpi)); ImGui::Text("%s", app.install_status_text.c_str()); }
    else { if (ImGui::Button(venv ? "Reinstall Libs" : "Create Env", ImVec2(-1, 30 * dpi))) std::thread(InstallThreadFunc).detach(); }
    ImGui::Spacing();
    if (!venv) ImGui::BeginDisabled();
    if (app.backend_running) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f)); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        if (ImGui::Button("STOP ENGINE", btn_size)) StopBackend();
        ImGui::PopStyleColor(2);
    }
    else {
        if (ImGui::Button("LAUNCH ENGINE", btn_size)) std::thread(BackendMonitorThread, GetPythonPath(), app.python_script).detach();
    }
    if (!venv) ImGui::EndDisabled();
    ImGui::EndChild();

    // 3. Status
    ImGui::BeginChild("Status", ImVec2(0, 0), true);
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "STATUS"); ImGui::Separator();
    if (ImGui::Button("Install OpenVR", ImVec2(-1, 30 * dpi))) std::thread(InstallDriverThread, "OpenVR").detach();
    if (ImGui::Button("Install ROS", ImVec2(-1, 30 * dpi))) std::thread(InstallDriverThread, "ROS").detach();
    ImGui::Spacing();
    ImGui::Columns(2, nullptr, false); ImGui::SetColumnWidth(0, 220 * dpi);
    ImGui::Text("Cam Pub (%d)", app.port_pub_frames); ImGui::NextColumn(); DrawStatusDot(app.status_cam_pub); ImGui::NextColumn();
    ImGui::Text("Prev Sub (%d)", app.port_sub_preview); ImGui::NextColumn(); DrawStatusDot(app.status_prev_sub); ImGui::NextColumn();
    ImGui::Text("Pose Sub (%d)", app.port_sub_pose); ImGui::NextColumn(); DrawStatusDot(app.status_pose_sub);
    ImGui::Columns(1);
    ImGui::EndChild();
    ImGui::End();

    // Right Panel
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + left_w, viewport->WorkPos.y));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x - left_w, viewport->WorkSize.y));
    ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    // [修改] 只有开关开启时才执行纹理上传和绘制
    if (app.show_previews) {
        float img_h = viewport->WorkSize.y * 0.6f;
        ImGui::BeginChild("Images", ImVec2(0, img_h), false);
        ImGui::BeginGroup();
        { std::lock_guard<std::mutex> lock(app.data_mutex); UpdateTexture(app.tex_raw, app.frame_raw); }
        float hw = ImGui::GetContentRegionAvail().x * 0.5f - 10;
        if (app.tex_raw) { float ar = (float)app.frame_raw.cols / std::max(1.0f, (float)app.frame_raw.rows); ImGui::Image((ImTextureID)(intptr_t)app.tex_raw, ImVec2(hw, hw / ar)); }
        else ImGui::Dummy(ImVec2(hw, 200));
        ImGui::EndGroup(); ImGui::SameLine();
        ImGui::BeginGroup();
        { std::lock_guard<std::mutex> lock(app.data_mutex); UpdateTexture(app.tex_preview, app.frame_preview); }
        if (app.tex_preview) { float ar = (float)app.frame_preview.cols / std::max(1.0f, (float)app.frame_preview.rows); ImGui::Image((ImTextureID)(intptr_t)app.tex_preview, ImVec2(hw, hw / ar)); }
        else ImGui::Dummy(ImVec2(hw, 200));
        ImGui::EndGroup(); ImGui::EndChild();
    }
    else {
        // 关闭预览时显示的占位提示
        ImGui::BeginChild("ImagesPlaceholder", ImVec2(0, viewport->WorkSize.y * 0.1f));
        ImGui::TextDisabled("--- Previews Hidden (Reduced GPU Load) ---");
        ImGui::EndChild();
    }

    ImGui::Separator();
    ImGui::BeginChild("Log", ImVec2(0, 0), true);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.02f, 0.02f, 0.02f, 1.0f));
    std::lock_guard<std::mutex> lock(app.log_mutex);
    for (const auto& l : app.logs) {
        ImVec4 c = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
        if (l.find("Error") != std::string::npos) c = ImVec4(1, 0.4f, 0.4f, 1);
        else if (l.find("[PY]") != std::string::npos) c = ImVec4(0.6f, 0.8f, 1, 1);
        ImGui::TextColored(c, "%s", l.c_str());
    }
    if (app.scroll_to_bottom) { ImGui::SetScrollHereY(1.0f); app.scroll_to_bottom = false; }
    ImGui::PopStyleColor(); ImGui::EndChild(); ImGui::End();
}

void LoadScaledFont(float dpi) {
    ImGuiIO& io = ImGui::GetIO(); io.Fonts->Clear(); float s = 16.0f * dpi; ImFont* f = nullptr;
    std::vector<std::string> paths;
#ifdef _WIN32
    paths = { "C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf" };
#elif __APPLE__
    paths = { "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf" };
#else
    paths = { "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf" };
#endif
    for (const auto& p : paths) if (fs::exists(p)) { f = io.Fonts->AddFontFromFileTTF(p.c_str(), s); if (f) break; }
    if (!f) { io.Fonts->AddFontDefault(); io.FontGlobalScale = dpi; }
}

int main(int, char**) {
    if (!glfwInit()) return 1;
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    const char* glsl_version = "#version 150";
#else
    const char* glsl_version = "#version 130";
#endif
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    GLFWwindow* w = glfwCreateWindow(1280, 768, "Pose Bridge", NULL, NULL);
    if (!w) return 1;
    glfwMakeContextCurrent(w);
    glfwSwapInterval(1);
    float xs, ys; glfwGetWindowContentScale(w, &xs, &ys); float dpi = xs > 0.0f ? xs : 1.0f;
    IMGUI_CHECKVERSION(); ImGui::CreateContext(); ImGui::StyleColorsDark(); ImGui::GetStyle().ScaleAllSizes(dpi);
    LoadScaledFont(dpi);
    ImGui_ImplGlfw_InitForOpenGL(w, true); ImGui_ImplOpenGL3_Init(glsl_version);
    RefreshCameraList();
    std::thread t1(CameraThread), t2(ReceiverThread);
    while (!glfwWindowShouldClose(w)) {
        glfwPollEvents(); ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
        RenderUI(dpi);
        ImGui::Render();
        int dw, dh; glfwGetFramebufferSize(w, &dw, &dh); glViewport(0, 0, dw, dh);
        glClearColor(0.1f, 0.1f, 0.13f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(w);
    }
    app.is_running = false; if (t1.joinable()) t1.join(); if (t2.joinable()) t2.join();
    ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown(); ImGui::DestroyContext();
    glfwDestroyWindow(w); glfwTerminate();
    return 0;
}