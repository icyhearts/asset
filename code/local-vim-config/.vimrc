let NERDTreeIgnore = ['\.pyc$', '^build$[[dir]]', '^__pycache__$[[dir]]', '\.swp$']
autocmd FileType python set expandtab tabstop=2 shiftwidth=2

nnoremap <leader>d ?^\s*def <CR>
nnoremap <leader>c ?^\s*class <CR>
nnoremap <leader>D /^\s*def <CR>
nnoremap <leader>C /^\s*class <CR>

" press \f, then register f will container file name
nnoremap <leader>f :let @f=@%<CR>
set path+=/share/users/like/package/simo_conda_sglang
