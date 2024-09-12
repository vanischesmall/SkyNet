-- ~/nvim/lua/vanische/settings.lua

vim.opt.cursorline = true
vim.opt.termguicolors = true
vim.opt.laststatus = 0
vim.opt.guicursor = "i:hor20-Underline"
vim.opt.tabstop = 4
vim.opt.shiftwidth = 4
vim.opt.encoding = "UTF-8"
vim.opt.mouse = "a"
vim.opt.termguicolors = true

vim.keymap.set("n", ";", ":", { noremap = true, silent = false })
vim.keymap.set("v", ";", ":", { noremap = true, silent = false })

vim.o.background = "dark"
