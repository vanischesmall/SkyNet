# Rice Aliases
alias pf "pfetch"
alias nf "neofetch"
alias cm "cmatrix"
alias cb "cbonsai --live --infinite"

# Utils Aliases 
alias vim "nvim"

alias md "mkdir -p"
alias rm "rm -rf"

alias ls  "exa --icons"
alias l   "exa --icons --long"
alias la  "exa --icons --long --all"
alias lt  "exa --icons --tree"
alias llt "exa --icons --long --tree"
alias lsu "cyme"
#alias l   "exa --icons --long --no-permissions --no-filesize --no-user --no-time"

alias ubu  "distrobox-enter Ubuntu-22.04"
alias ros2 "cd ~/ros2_ws; distrobox-enter Ubuntu-22.04"


set -x QT_QPA_PLATFORM xcb


function fish_prompt -d "Write out the prompt"
    printf '%s@%s %s%s%s > ' $USER $hostname \
        (set_color $fish_color_cwd) (prompt_pwd) (set_color normal)
end

if status is-interactive
    set fish_greeting
end

starship init fish | source
cat ~/.cache/ags/user/generated/terminal/sequences.txt

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
if test -f /home/vanische/.miniconda3/bin/conda
    eval /home/vanische/.miniconda3/bin/conda "shell.fish" "hook" $argv | source
else
    if test -f "/home/vanische/.miniconda3/etc/fish/conf.d/conda.fish"
        . "/home/vanische/.miniconda3/etc/fish/conf.d/conda.fish"
    else
        set -x PATH "/home/vanische/.miniconda3/bin" $PATH
    end
end
# <<< conda initialize <<<
