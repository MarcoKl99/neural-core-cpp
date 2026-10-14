## Directory Access Restrictions

- Only access, read, create, modify, or delete files and directories within the current project directory.
- Never access files or directories outside the project root, including parent directories, home directories, system directories, or other projects.
- Do not use shell commands, scripts, or other tools to read or modify files outside the project directory.
- Do not follow symbolic links that point outside the project directory.
- If a task requires access to files outside the project directory, stop and ask the user for explicit permission before proceeding.
- Treat these restrictions as mandatory and do not bypass them under any circumstances.
