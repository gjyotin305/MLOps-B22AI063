import { cn } from "@/lib/utils";

interface MarkdownViewerProps {
  content: string;
}

export const MarkdownViewer = ({ content }: MarkdownViewerProps) => {
  // Simple markdown parsing - handles common elements
  const parseMarkdown = (text: string) => {
    const lines = text.split("\n");
    const elements: JSX.Element[] = [];
    let inCodeBlock = false;
    let codeContent: string[] = [];
    let codeLanguage = "";
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Code block start/end
      if (line.startsWith("```")) {
        if (inCodeBlock) {
          elements.push(
            <pre
              key={elements.length}
              className="bg-muted p-4 rounded-lg overflow-x-auto my-4 text-sm"
            >
              <code className="text-foreground">{codeContent.join("\n")}</code>
            </pre>
          );
          codeContent = [];
          inCodeBlock = false;
        } else {
          inCodeBlock = true;
          codeLanguage = line.slice(3);
        }
        i++;
        continue;
      }

      if (inCodeBlock) {
        codeContent.push(line);
        i++;
        continue;
      }

      // Check for table (table starts with | and has header separator on next line)
      if (line.trim().startsWith("|") && i + 1 < lines.length) {
        const nextLine = lines[i + 1];
        if (nextLine.trim().startsWith("|") && /^[\s\|\-:]+$/.test(nextLine)) {
          // Parse table
          const tableRows: string[][] = [];
          let j = i;
          
          // Parse all table rows
          while (j < lines.length && lines[j].trim().startsWith("|")) {
            const cells = lines[j]
              .split("|")
              .slice(1, -1)
              .map(cell => cell.trim());
            tableRows.push(cells);
            j++;
          }

          // Skip separator row and render table
          if (tableRows.length > 1) {
            const headerRow = tableRows[0];
            const bodyRows = tableRows.slice(2); // Skip header and separator

            elements.push(
              <div key={elements.length} className="overflow-x-auto my-4">
                <table className="w-full border-collapse border border-border">
                  <thead>
                    <tr className="bg-muted">
                      {headerRow.map((cell, idx) => (
                        <th
                          key={idx}
                          className="border border-border px-4 py-2 text-left font-semibold text-foreground"
                        >
                          {parseInline(cell)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {bodyRows.map((row, rowIdx) => (
                      <tr key={rowIdx} className="hover:bg-muted/50">
                        {row.map((cell, cellIdx) => (
                          <td
                            key={cellIdx}
                            className="border border-border px-4 py-2 text-foreground"
                          >
                            {parseInline(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );

            i = j;
            continue;
          }
        }
      }

      // Empty line
      if (!line.trim()) {
        elements.push(<div key={elements.length} className="h-4" />);
        i++;
        continue;
      }

      // Headers
      if (line.startsWith("# ")) {
        elements.push(
          <h1 key={elements.length} className="text-3xl font-bold mt-6 mb-4 text-foreground">
            {parseInline(line.slice(2))}
          </h1>
        );
        i++;
        continue;
      }
      if (line.startsWith("## ")) {
        elements.push(
          <h2 key={elements.length} className="text-2xl font-semibold mt-5 mb-3 text-foreground">
            {parseInline(line.slice(3))}
          </h2>
        );
        i++;
        continue;
      }
      if (line.startsWith("### ")) {
        elements.push(
          <h3 key={elements.length} className="text-xl font-semibold mt-4 mb-2 text-foreground">
            {parseInline(line.slice(4))}
          </h3>
        );
        i++;
        continue;
      }

      // Bullet points
      if (line.match(/^[\-\*]\s/)) {
        elements.push(
          <li key={elements.length} className="ml-4 list-disc text-foreground">
            {parseInline(line.slice(2))}
          </li>
        );
        i++;
        continue;
      }

      // Numbered list
      if (line.match(/^\d+\.\s/)) {
        elements.push(
          <li key={elements.length} className="ml-4 list-decimal text-foreground">
            {parseInline(line.replace(/^\d+\.\s/, ""))}
          </li>
        );
        i++;
        continue;
      }

      // Blockquote
      if (line.startsWith("> ")) {
        elements.push(
          <blockquote
            key={elements.length}
            className="border-l-4 border-primary pl-4 italic text-muted-foreground my-2"
          >
            {parseInline(line.slice(2))}
          </blockquote>
        );
        i++;
        continue;
      }

      // Regular paragraph
      elements.push(
        <p key={elements.length} className="text-foreground leading-relaxed">
          {parseInline(line)}
        </p>
      );
      i++;
    }

    return elements;
  };

  const parseInline = (text: string): React.ReactNode => {
    // Split and process inline elements
    const parts: React.ReactNode[] = [];
    let remaining = text;
    let key = 0;

    while (remaining.length > 0) {
      // Inline code
      const codeMatch = remaining.match(/`([^`]+)`/);
      if (codeMatch && codeMatch.index !== undefined) {
        if (codeMatch.index > 0) {
          parts.push(parseInlineStyles(remaining.slice(0, codeMatch.index), key++));
        }
        parts.push(
          <code
            key={key++}
            className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono text-primary"
          >
            {codeMatch[1]}
          </code>
        );
        remaining = remaining.slice(codeMatch.index + codeMatch[0].length);
        continue;
      }

      // Link
      const linkMatch = remaining.match(/\[([^\]]+)\]\(([^)]+)\)/);
      if (linkMatch && linkMatch.index !== undefined) {
        if (linkMatch.index > 0) {
          parts.push(parseInlineStyles(remaining.slice(0, linkMatch.index), key++));
        }
        parts.push(
          <a
            key={key++}
            href={linkMatch[2]}
            className="text-primary underline hover:no-underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            {linkMatch[1]}
          </a>
        );
        remaining = remaining.slice(linkMatch.index + linkMatch[0].length);
        continue;
      }

      // No more special patterns, process rest with inline styles
      parts.push(parseInlineStyles(remaining, key++));
      break;
    }

    return parts;
  };

  const parseInlineStyles = (text: string, key: number): React.ReactNode => {
    // Bold
    let result: React.ReactNode = text;
    
    if (text.includes("**")) {
      const parts = text.split(/\*\*([^*]+)\*\*/);
      result = parts.map((part, i) =>
        i % 2 === 1 ? (
          <strong key={`${key}-bold-${i}`} className="font-semibold">
            {part}
          </strong>
        ) : (
          part
        )
      );
    }

    return <span key={key}>{result}</span>;
  };

  return (
    <div className="prose prose-slate dark:prose-invert max-w-none">
      {parseMarkdown(content)}
    </div>
  );
};
