const vscode = require("vscode");
const { exec } = require("child_process");
const slugify = require("slugify");

function activate(context) {
  // 기존의 명령어 등록
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

  // 새로운 명령어 등록
  const disposableSlug = vscode.commands.registerCommand(
    "extension.addSlugsToHeadings",
    function () {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor found.");
        return;
      }

      const document = editor.document;
      const text = document.getText();
      const lines = text.split("\n");

      const edits = [];

      for (let i = 0; i < lines.length; i++) {
        let line = lines[i];

        // 코드 블록 내부는 무시
        if (isInsideCodeBlock(lines, i)) {
          continue;
        }

        // 헤딩 검사 (예: ## 헤딩)
        const headingRegex = /^(#{1,6})\s+(.*?)(\s*\{#.*\})?$/;
        const match = headingRegex.exec(line);

        if (match) {
          const hashes = match[1];
          const headingText = match[2];
          const existingSlug = match[3];

          if (!existingSlug) {
            // 슬러그 생성
            const slug = generateSlug(headingText);

            // 슬러그 추가
            const newLine = `${hashes} ${headingText} {#${slug}}`;

            // 편집 내용 저장
            const position = new vscode.Position(i, 0);
            const lineLength = lines[i].length;
            const range = new vscode.Range(
              position,
              new vscode.Position(i, lineLength)
            );

            edits.push({ range: range, newText: newLine });
          }
        }
      }

      if (edits.length > 0) {
        editor
          .edit((editBuilder) => {
            edits.forEach((edit) => {
              editBuilder.replace(edit.range, edit.newText);
            });
          })
          .then(() => {
            vscode.window.showInformationMessage(
              "헤딩에 슬러그가 추가되었습니다."
            );
          });
      } else {
        vscode.window.showInformationMessage("추가할 슬러그가 없습니다.");
      }
    }
  );

  context.subscriptions.push(disposable);
  context.subscriptions.push(disposableSlug);
}

function isInsideCodeBlock(lines, currentIndex) {
  let insideCodeBlock = false;
  for (let i = 0; i <= currentIndex; i++) {
    const line = lines[i];
    if (line.trim().startsWith("```")) {
      insideCodeBlock = !insideCodeBlock;
    }
  }
  return insideCodeBlock;
}

function generateSlug(text) {
  return slugify(text, {
    lower: true,
    remove: /[^\w\s_-]/g, // `_`, `-`, 공백, 알파벳, 숫자만 허용
    strict: false,
  });
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
