-- =============================================================================
-- nvim-dap 配置: debugpy attach 到子进程 (支持多 worker)
--
-- 把这段代码放到你的 nvim 配置中:
--   ~/.config/nvim/lua/plugins/dap.lua   (lazy.nvim 用户)
--   或 ~/.config/nvim/init.lua           (直接写)
--   或 :luafile 这个文件来临时加载
-- =============================================================================

local dap = require("dap")

-- ========== 1. 配置 Python adapter (debugpy) ==========
-- 如果你已经用 mason.nvim 安装了 debugpy, 路径类似:
--   ~/.local/share/nvim/mason/packages/debugpy/venv/bin/python
-- 如果你用 conda 里的 debugpy, 就直接指向 conda python:
dap.adapters.python_attach = {
    type = "server",
    host = "127.0.0.1",
    port = 5678,               -- 默认端口, 会被 configuration 的 connect.port 覆盖
}

-- 也可以配置 launch 模式 (直接启动进程):
dap.adapters.python_launch = {
    type = "executable",
    -- ↓↓↓ 改成你的 conda 环境中的 python ↓↓↓
    command = "/share_data/users/like/miniconda3/envs/simo_sglang/bin/python",
    args = { "-m", "debugpy.adapter" },
}

-- ========== 2. 配置 debug configurations ==========
dap.configurations.python = {
    -- [配置 A] attach Worker-0 (port 5678)
    {
        type = "python_attach",
        request = "attach",
        name = "Attach to Worker-0 (port 5678)",
        connect = {
            host = "127.0.0.1",
            port = 5678,
        },
        pathMappings = {
            {
                localRoot = vim.fn.getcwd(),
                remoteRoot = vim.fn.getcwd(),
            },
        },
    },

    -- [配置 B] attach Worker-1 (port 5679)
    {
        type = "python_attach",
        request = "attach",
        name = "Attach to Worker-1 (port 5679)",
        connect = {
            host = "127.0.0.1",
            port = 5679,
        },
        pathMappings = {
            {
                localRoot = vim.fn.getcwd(),
                remoteRoot = vim.fn.getcwd(),
            },
        },
    },

    -- [配置 C] 动态输入端口 (通用方案, 适合 worker 数量不固定)
    {
        type = "python_attach",
        request = "attach",
        name = "Attach to Worker (input port)",
        connect = function()
            local port = tonumber(vim.fn.input("Enter debugpy port: ", "5678"))
            return {
                host = "127.0.0.1",
                port = port,
            }
        end,
        pathMappings = {
            {
                localRoot = vim.fn.getcwd(),
                remoteRoot = vim.fn.getcwd(),
            },
        },
    },

    -- [配置 D] 直接 launch 主进程 (不调试子进程)
    {
        type = "python_launch",
        request = "launch",
        name = "Launch main_server.py",
        program = "${workspaceFolder}/like-useful/demo_debugpy/main_server.py",
        args = { "--num-workers", "2" },
        console = "integratedTerminal",
    },

    -- [配置 E] attach 到 SGLang TP Worker-0 (实际使用)
    {
        type = "python_attach",
        request = "attach",
        name = "SGLang TP Worker-0 (port 5678)",
        connect = {
            host = "127.0.0.1",
            port = 5678,
        },
        pathMappings = {
            {
                localRoot = "/share_data/users/like/package/h100/package/sglang_kernel_src",
                remoteRoot = "/share_data/users/like/package/h100/package/sglang_kernel_src",
            },
        },
    },

    -- [配置 F] attach 到 SGLang TP Worker-1 (实际使用)
    {
        type = "python_attach",
        request = "attach",
        name = "SGLang TP Worker-1 (port 5679)",
        connect = {
            host = "127.0.0.1",
            port = 5679,
        },
        pathMappings = {
            {
                localRoot = "/share_data/users/like/package/h100/package/sglang_kernel_src",
                remoteRoot = "/share_data/users/like/package/h100/package/sglang_kernel_src",
            },
        },
    },
}

-- ========== 3. 快捷键 ==========
vim.keymap.set("n", "<leader>dc", function() dap.continue() end,          { desc = "DAP Continue" })
vim.keymap.set("n", "<leader>db", function() dap.toggle_breakpoint() end,  { desc = "DAP Breakpoint" })
vim.keymap.set("n", "<leader>dn", function() dap.step_over() end,         { desc = "DAP Step Over" })
vim.keymap.set("n", "<leader>di", function() dap.step_into() end,         { desc = "DAP Step Into" })
vim.keymap.set("n", "<leader>do", function() dap.step_out() end,          { desc = "DAP Step Out" })
vim.keymap.set("n", "<leader>dr", function() dap.repl.open() end,         { desc = "DAP REPL" })
vim.keymap.set("n", "<leader>dl", function() dap.run_last() end,          { desc = "DAP Run Last" })
vim.keymap.set("n", "<leader>dt", function() dap.terminate() end,         { desc = "DAP Terminate" })

-- ========== 4. (可选) dap-ui ==========
local ok, dapui = pcall(require, "dapui")
if ok then
    dapui.setup()
    dap.listeners.after.event_initialized["dapui_config"] = function() dapui.open() end
    dap.listeners.before.event_terminated["dapui_config"] = function() dapui.close() end
    dap.listeners.before.event_exited["dapui_config"]     = function() dapui.close() end
end

print("[dap_config.lua] nvim-dap python configurations loaded")
