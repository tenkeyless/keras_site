const vscode = require("vscode");
const { exec } = require("child_process");

function activate(context) {
  const disposable = vscode.commands.registerCommand(
    "extension.insertMarkdownOutput",
    function () {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor found.");
        return;
      }

      // 현재 열려 있는 파일 경로
      const filePath = editor.document.uri.fsPath;

      // 확장 디렉토리 내 Python 스크립트 경로
      const extensionPath = context.extensionPath; // 확장 경로 가져오기
      const scriptPath = `${extensionPath}/fetch_markdown.py`;
      const tempFile = "/tmp/output.md";
      const command = `python ${scriptPath} ${filePath} ${tempFile}`;

      exec(command, (error, stdout, stderr) => {
        if (error) {
          vscode.window.showErrorMessage(`Error: ${stderr}`);
          return;
        }

        // Markdown 파일 읽기
        const fs = require("fs");
        const markdownContent = fs.readFileSync(tempFile, "utf8");

        // 활성 에디터에 삽입
        editor.edit((editBuilder) => {
          const position = editor.selection.active;
          editBuilder.insert(position, markdownContent);
        });
      });
    }
  );

  context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
