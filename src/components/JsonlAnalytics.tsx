import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

interface JsonlAnalyticsProps {
  data: object[];
}

const COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"];

export const JsonlAnalytics = ({ data }: JsonlAnalyticsProps) => {
  const analytics = useMemo(() => {
    if (!data || data.length === 0) return null;

    // Extract all numeric and categorical fields
    const numericFields = new Map<string, number[]>();
    const categoricalFields = new Map<string, Map<string, number>>();
    const recordCount = data.length;

    data.forEach((record) => {
      if (typeof record !== "object" || record === null) return;

      Object.entries(record).forEach(([key, value]) => {
        if (typeof value === "number") {
          if (!numericFields.has(key)) {
            numericFields.set(key, []);
          }
          numericFields.get(key)!.push(value);
        } else if (typeof value === "string") {
          if (!categoricalFields.has(key)) {
            categoricalFields.set(key, new Map());
          }
          const counts = categoricalFields.get(key)!;
          counts.set(value, (counts.get(value) || 0) + 1);
        }
      });
    });

    // Calculate statistics for numeric fields
    const numericStats = new Map<string, { mean: number; min: number; max: number; median: number }>();
    numericFields.forEach((values, key) => {
      const sorted = [...values].sort((a, b) => a - b);
      const sum = values.reduce((a, b) => a + b, 0);
      const mean = sum / values.length;
      const median = sorted[Math.floor(sorted.length / 2)];
      const min = Math.min(...values);
      const max = Math.max(...values);
      numericStats.set(key, { mean, min, max, median });
    });

    return {
      recordCount,
      numericFields,
      categoricalFields,
      numericStats,
    };
  }, [data]);

  if (!analytics) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-muted-foreground">No data available for analytics</p>
        </CardContent>
      </Card>
    );
  }

  const { recordCount, numericFields, categoricalFields, numericStats } = analytics;

  // Determine numeric fields to display
  const numericFieldsToDisplay = Array.from(numericStats.entries())
    .filter(([key]) => 
      key.includes("time") || 
      key.includes("GFLOPs") || 
      key.includes("accuracy") ||
      key.includes("Params") ||
      key.includes("flops") ||
      key.includes("gflop")
    )
    .slice(0, 5);

  // Categorical fields to display
  const categoriesToDisplay = ["model", "dataset", "kernel", "optimizer"];
  const categoricalCharts = Array.from(categoricalFields.entries())
    .filter(([key]) => categoriesToDisplay.some(cat => key.toLowerCase().includes(cat)))
    .slice(0, 3);

  return (
    <div className="space-y-6">
      {/* Summary Statistics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Summary Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Total Records</p>
              <p className="text-2xl font-bold text-foreground">{recordCount}</p>
            </div>
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Numeric Fields</p>
              <p className="text-2xl font-bold text-foreground">{numericFields.size}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Numeric Field Statistics */}
      {numericFieldsToDisplay.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Numeric Field Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 px-3 font-semibold text-foreground">Field</th>
                    <th className="text-right py-2 px-3 font-semibold text-foreground">Min</th>
                    <th className="text-right py-2 px-3 font-semibold text-foreground">Max</th>
                    <th className="text-right py-2 px-3 font-semibold text-foreground">Mean</th>
                    <th className="text-right py-2 px-3 font-semibold text-foreground">Median</th>
                  </tr>
                </thead>
                <tbody>
                  {numericFieldsToDisplay.map(([field, stats]) => (
                    <tr key={field} className="border-b hover:bg-muted/50">
                      <td className="py-2 px-3 text-foreground font-medium">{field}</td>
                      <td className="text-right py-2 px-3 text-muted-foreground">{stats.min.toFixed(2)}</td>
                      <td className="text-right py-2 px-3 text-muted-foreground">{stats.max.toFixed(2)}</td>
                      <td className="text-right py-2 px-3 text-muted-foreground">{stats.mean.toFixed(2)}</td>
                      <td className="text-right py-2 px-3 text-muted-foreground">{stats.median.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Categorical Distributions */}
      {categoricalCharts.map(([field, counts]) => {
        const chartData = Array.from(counts.entries()).map(([category, count]) => ({
          name: category,
          value: count,
        }));

        return (
          <Card key={field}>
            <CardHeader>
              <CardTitle className="text-lg capitalize">{field} Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                {chartData.length <= 5 ? (
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {chartData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                ) : (
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#3b82f6" />
                  </BarChart>
                )}
              </ResponsiveContainer>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
};
