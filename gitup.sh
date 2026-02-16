#!/bin/bash

# 1. Add all changes (including untracked files)
git add .

# 2. Ask for a commit message
echo "Enter your commit message:"
read commit_msg

# 3. Commit with that message
git commit -m "$commit_msg"

# 4. Push to the current branch (HEAD targets whatever branch you are on)
git push origin HEAD
