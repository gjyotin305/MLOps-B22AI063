import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ArrowLeft, FileText, Database } from "lucide-react";
import { JsonlViewer } from "@/components/JsonlViewer";
import { MarkdownViewer } from "@/components/MarkdownViewer";

const AssignmentDetail = () => {
  const { assignmentId } = useParams<{ assignmentId: string }>();
  const [readme, setReadme] = useState<string>("");
  const [jsonlData, setJsonlData] = useState<object[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Fetch README.md
        const readmeRes = await fetch(`/${assignmentId}/README.md`);
        if (readmeRes.ok) {
          const readmeText = await readmeRes.text();
          setReadme(readmeText);
        } else {
          setReadme("*No README.md found*");
        }

        // Fetch samples.jsonl
        const jsonlRes = await fetch(`/${assignmentId}/samples.jsonl`);
        if (jsonlRes.ok) {
          const jsonlText = await jsonlRes.text();
          const lines = jsonlText.trim().split("\n").filter(line => line.trim());
          const parsed = lines.map((line, index) => {
            try {
              return JSON.parse(line);
            } catch {
              return { _parseError: true, _line: index + 1, _raw: line };
            }
          });
          setJsonlData(parsed);
        } else {
          setJsonlData([]);
        }
      } catch (err) {
        setError("Failed to load assignment data");
      } finally {
        setLoading(false);
      }
    };

    if (assignmentId) {
      fetchData();
    }
  }, [assignmentId]);

  const formatAssignmentName = (id: string) => {
    return id.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">Loading assignment...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-6">
          <Link to="/">
            <Button variant="ghost" size="sm" className="mb-4">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Assignments
            </Button>
          </Link>
          <h1 className="text-3xl font-bold text-foreground">
            {formatAssignmentName(assignmentId || "")}
          </h1>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {error ? (
          <Card className="border-destructive">
            <CardContent className="pt-6 text-destructive">{error}</CardContent>
          </Card>
        ) : (
          <Tabs defaultValue="readme" className="space-y-4">
            <TabsList>
              <TabsTrigger value="readme" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                README
              </TabsTrigger>
              <TabsTrigger value="jsonl" className="flex items-center gap-2">
                <Database className="h-4 w-4" />
                JSONL Data ({jsonlData.length} rows)
              </TabsTrigger>
            </TabsList>

            <TabsContent value="readme">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    README.md
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <MarkdownViewer content={readme} />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="jsonl">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Database className="h-5 w-5" />
                    samples.jsonl
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <JsonlViewer data={jsonlData} />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}
      </main>
    </div>
  );
};

export default AssignmentDetail;
