import * as vscode from "vscode";
import { LanguageClient, LanguageClientOptions, ServerOptions } from "vscode-languageclient/node";
import { exec } from "child_process";

export async function activate(_context: vscode.ExtensionContext) {
  exec("ocimatic version", (err) => {
    if (err) {
      vscode.window.showWarningMessage(
        "Ocimatic command not found. Please install 'ocimatic' and ensure it is in your PATH."
      );
      return;
    }

    // Server options: run 'ocimatic lsp'
    const serverOptions: ServerOptions = {
      command: "ocimatic",
      args: ["lsp"],
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
      documentSelector: [{ scheme: "file", language: "ocimatic-testplan" }],
    };

    // Create and start the language client
    const client = new LanguageClient(
      "ocimatic-language-server",
      "Ocimatic",
      serverOptions,
      clientOptions
    );

    client.start();
  });
}
