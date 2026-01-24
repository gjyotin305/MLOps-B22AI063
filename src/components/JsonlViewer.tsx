import { useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ChevronDown, ChevronRight, Search, Copy, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface JsonlViewerProps {
  data: object[];
}

interface JsonNodeProps {
  data: unknown;
  depth?: number;
  isLast?: boolean;
}

const JsonNode = ({ data, depth = 0, isLast = true }: JsonNodeProps) => {
  const [expanded, setExpanded] = useState(depth < 2);

  if (data === null) {
    return <span className="text-muted-foreground italic">null</span>;
  }

  if (typeof data === "boolean") {
    return <span className="text-accent-foreground font-medium">{data.toString()}</span>;
  }

  if (typeof data === "number") {
    return <span className="text-primary">{data}</span>;
  }

  if (typeof data === "string") {
    const truncated = data.length > 100;
    const displayText = truncated ? data.slice(0, 100) + "..." : data;
    return (
      <span className="text-muted-foreground">
        "{displayText}"
        {truncated && (
          <Badge variant="outline" className="ml-1 text-xs">
            +{data.length - 100} chars
          </Badge>
        )}
      </span>
    );
  }

  if (Array.isArray(data)) {
    if (data.length === 0) {
      return <span className="text-muted-foreground">[]</span>;
    }

    return (
      <span>
        <button
          onClick={() => setExpanded(!expanded)}
          className="inline-flex items-center hover:bg-muted rounded px-1"
        >
          {expanded ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
          <span className="text-muted-foreground ml-1">
            [{data.length} items]
          </span>
        </button>
        {expanded && (
          <div className="ml-4 border-l border-border pl-2">
            {data.map((item, index) => (
              <div key={index} className="py-0.5">
                <span className="text-muted-foreground mr-2">{index}:</span>
                <JsonNode
                  data={item}
                  depth={depth + 1}
                  isLast={index === data.length - 1}
                />
              </div>
            ))}
          </div>
        )}
      </span>
    );
  }

  if (typeof data === "object") {
    const entries = Object.entries(data);
    if (entries.length === 0) {
      return <span className="text-muted-foreground">{"{}"}</span>;
    }

    return (
      <span>
        <button
          onClick={() => setExpanded(!expanded)}
          className="inline-flex items-center hover:bg-muted rounded px-1"
        >
          {expanded ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
          <span className="text-muted-foreground ml-1">
            {"{"}
            {entries.length} keys{"}"}
          </span>
        </button>
        {expanded && (
          <div className="ml-4 border-l border-border pl-2">
            {entries.map(([key, value], index) => (
              <div key={key} className="py-0.5">
                <span className="text-primary font-medium">
                  "{key}"
                </span>
                <span className="text-muted-foreground">: </span>
                <JsonNode
                  data={value}
                  depth={depth + 1}
                  isLast={index === entries.length - 1}
                />
              </div>
            ))}
          </div>
        )}
      </span>
    );
  }

  return <span>{String(data)}</span>;
};

const ITEMS_PER_PAGE = 20;

export const JsonlViewer = ({ data }: JsonlViewerProps) => {
  const [search, setSearch] = useState("");
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  const filteredData = data.filter((row) => {
    if (!search) return true;
    return JSON.stringify(row).toLowerCase().includes(search.toLowerCase());
  });

  const totalPages = Math.ceil(filteredData.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const paginatedData = filteredData.slice(startIndex, startIndex + ITEMS_PER_PAGE);

  // Reset to page 1 when search changes
  const handleSearchChange = (value: string) => {
    setSearch(value);
    setCurrentPage(1);
  };

  const copyRow = (row: object) => {
    navigator.clipboard.writeText(JSON.stringify(row, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (data.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No JSONL data found
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search in data..."
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="pl-9"
          />
        </div>
        <Badge variant="secondary">
          {filteredData.length} / {data.length} rows
        </Badge>
      </div>

      <ScrollArea className="h-[600px] rounded-md border">
        <div className="p-4 space-y-2">
          {paginatedData.map((row, index) => {
            const actualIndex = startIndex + index;
            const hasError = (row as any)._parseError;
            return (
              <div
                key={actualIndex}
                className={cn(
                  "p-3 rounded-lg border font-mono text-sm transition-colors",
                  hasError
                    ? "border-destructive bg-destructive/5"
                    : "border-border hover:border-primary/50",
                  selectedRow === actualIndex && "ring-2 ring-primary"
                )}
                onClick={() => setSelectedRow(actualIndex)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 overflow-x-auto">
                    <span className="text-muted-foreground mr-2 select-none">
                      #{actualIndex + 1}
                    </span>
                    {hasError ? (
                      <span className="text-destructive">
                        Parse error on line {(row as any)._line}:{" "}
                        {(row as any)._raw}
                      </span>
                    ) : (
                      <JsonNode data={row} />
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="shrink-0 h-8 w-8"
                    onClick={(e) => {
                      e.stopPropagation();
                      copyRow(row);
                    }}
                  >
                    {copied && selectedRow === actualIndex ? (
                      <Check className="h-4 w-4 text-primary" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {startIndex + 1}-{Math.min(startIndex + ITEMS_PER_PAGE, filteredData.length)} of {filteredData.length}
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
            >
              Previous
            </Button>
            <div className="flex items-center gap-1">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNum: number;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (currentPage <= 3) {
                  pageNum = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }
                return (
                  <Button
                    key={pageNum}
                    variant={currentPage === pageNum ? "default" : "outline"}
                    size="sm"
                    className="w-8 h-8 p-0"
                    onClick={() => setCurrentPage(pageNum)}
                  >
                    {pageNum}
                  </Button>
                );
              })}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
