import * as vscode from "vscode";
import { LanguageClient, LanguageClientOptions, ServerOptions } from "vscode-languageclient/node";

export async function activate(_context: vscode.ExtensionContext) {
  // Server options: run 'ocimatic lsp'
  const serverOptions: ServerOptions = {
    command: "ocimatic",
    args: ["lsp"],
  };

  // Client options
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "ocimatic-testplan" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.testplan.txt"),
    },
  };

  // Create and start the language client
  const client = new LanguageClient(
    "ocimatic-language-server",
    "Ocimatic",
    serverOptions,
    clientOptions
  );

  client.start();
}
