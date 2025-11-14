import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { Bot, User } from "lucide-react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import ReactMarkdown from "react-markdown";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ChartData {
  chart_type: string;
  data: Record<string, any>;
  format: string;
}

interface SourceEntry {
  source: string;
  page?: number;
  score?: number;
  excerpt?: string;
  section_heading?: string;
  clause?: string;
  clause_heading?: string;
}

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  response_type?: string;
  charts?: ChartData | null;
  chart?: any; // Wage schedule chart format
  chart_type?: string; // Wage schedule chart type
  results?: any[] | null;
  total_rows?: number;
  userQuery?: string; // Original user query to detect plot requests
  query_classification?: string; // SQL or Knowledge/General
  answer_points?: string[];
  disclaimer?: string | null;
  sources?: SourceEntry[] | null;
  opening?: string | null;
}

const ChatMessage = ({
  role,
  content,
  response_type,
  charts,
  chart,
  chart_type,
  results,
  total_rows,
  userQuery,
  query_classification,
  answer_points,
  disclaimer,
  sources,
  opening,
}: ChatMessageProps) => {
  // Extract average data for table display
  const getAverageData = (charts: ChartData) => {
    if (!charts?.data) return null;
    
    const { data } = charts;
    const averages: Record<string, number> = {};
    
    // Get all fields that start with "average_"
    Object.keys(data).forEach(key => {
      if (key.startsWith('average_') && typeof data[key] === 'number') {
        const label = key.replace('average_', '').replace(/_/g, ' ');
        averages[label] = data[key];
      }
    });
    
    return Object.keys(averages).length > 0 ? averages : null;
  };

  // Format chart data for recharts from charts object
  const formatChartDataFromCharts = (charts: ChartData) => {
    if (!charts?.data) return null;
    
    const { data } = charts;
    const dates = data.dates || [];
    
    // If no dates or only aggregate data, return null
    if (!dates.length) return null;
    
    // Get all trend fields (fields ending with _trend)
    const trendFields = Object.keys(data).filter(key => 
      key.endsWith('_trend') && Array.isArray(data[key])
    );
    
    // If no trend data, don't show chart
    if (trendFields.length === 0) return null;
    
    // Transform the data into recharts format
    return dates.map((date: string, index: number) => {
      const point: any = { date };
      
      // Add all trend data series
      trendFields.forEach(key => {
        if (data[key][index] !== undefined) {
          // Create a clean label by removing _trend and formatting
          const label = key.replace('_trend', '').replace(/_/g, ' ');
          point[label] = data[key][index];
        }
      });
      
      return point;
    });
  };

  // Format wage schedule chart data (from backend chart prop)
  const formatWageScheduleChart = (chart: any, chartType: string) => {
    if (!chart || !chart.series) return null;
    
    if (chartType === "line") {
      // Line chart: multiple series with points
      const allYears = new Set<number>();
      chart.series.forEach((series: any) => {
        series.points?.forEach((point: any) => {
          if (point.fiscal_year) allYears.add(point.fiscal_year);
        });
      });
      
      const sortedYears = Array.from(allYears).sort((a, b) => a - b);
      
      return sortedYears.map(year => {
        const point: any = { date: year.toString() };
        chart.series.forEach((series: any) => {
          const seriesPoint = series.points?.find((p: any) => p.fiscal_year === year);
          if (seriesPoint) {
            point[series.name] = seriesPoint.value;
          }
        });
        return point;
      });
    } else {
      // Bar chart: single series with name/value pairs
      // Format as array with single object containing all values
      const barData: any = { date: "Comparison" };
      chart.series.forEach((item: any) => {
        barData[item.name] = item.value;
      });
      return [barData];
    }
  };

  // Format chart data from SQL results (if results have date column)
  const formatChartDataFromResults = (results: any[]) => {
    if (!results || results.length === 0) return null;
    
    // Check if results have FiscalYear column (for wage schedule)
    if (results[0].FiscalYear !== undefined) {
      const numericKeys = Object.keys(results[0]).filter(key => {
        if (key === 'FiscalYear') return false;
        if (key.toLowerCase().includes('id')) return false;
        if (key.toLowerCase().includes('date') && !key.toLowerCase().includes('fiscal')) return false;
        const value = results[0][key];
        return typeof value === 'number';
      });
      
      if (numericKeys.length === 0) return null;
      
      // Group by FiscalYear and average numeric columns
      const yearMap = new Map<number, any>();
      results.forEach((row: any) => {
        const year = row.FiscalYear;
        if (!yearMap.has(year)) {
          yearMap.set(year, { date: year.toString(), counts: {} });
        }
        const point = yearMap.get(year);
        numericKeys.forEach(key => {
          if (!point[key]) {
            point[key] = { sum: 0, count: 0 };
          }
          point[key].sum += row[key] || 0;
          point[key].count += 1;
        });
      });
      
      return Array.from(yearMap.values()).map(point => {
        const result: any = { date: point.date };
        numericKeys.forEach(key => {
          if (point[key]) {
            result[key.replace(/_/g, ' ')] = point[key].sum / point[key].count;
          }
        });
        return result;
      }).sort((a, b) => parseInt(a.date) - parseInt(b.date));
    }
    
    // Check if results have a date column
    const dateKeys = Object.keys(results[0]).filter(key => 
      key.toLowerCase().includes('date') || key.toLowerCase() === 'date'
    );
    
    if (dateKeys.length === 0) return null;
    
    const dateKey = dateKeys[0];
    
    // Get numeric columns (exclude date and id columns)
    const numericKeys = Object.keys(results[0]).filter(key => {
      if (key === dateKey) return false;
      if (key.toLowerCase().includes('id')) return false;
      const value = results[0][key];
      return typeof value === 'number';
    });
    
    if (numericKeys.length === 0) return null;
    
    // Transform results to chart format and sort by date ascending (oldest first)
    const chartData = results.map((row: any) => {
      const point: any = { 
        date: row[dateKey] || 'Unknown'
      };
      
      numericKeys.forEach(key => {
        point[key.replace(/_/g, ' ')] = row[key];
      });
      
      return point;
    });
    
    // Sort by date ascending (oldest to newest)
    chartData.sort((a, b) => {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      return dateA.getTime() - dateB.getTime();
    });
    
    return chartData;
  };

  const averageData = charts ? getAverageData(charts) : null;
  
  // Determine if we should show a chart
  // 1. Check if charts object has chart data
  // 2. Check if user asked for plot/chart
  // 3. Check if results have date-based data that can be charted
  // 4. Always show wage schedule charts if available
  const shouldShowChart = userQuery && (
    userQuery.toLowerCase().includes('plot') || 
    userQuery.toLowerCase().includes('chart') || 
    userQuery.toLowerCase().includes('graph') ||
    userQuery.toLowerCase().includes('visualize') ||
    userQuery.toLowerCase().includes('trend')
  );
  
  const chartDataFromCharts = charts ? formatChartDataFromCharts(charts) : null;
  const chartDataFromWageSchedule = chart ? formatWageScheduleChart(chart, chart_type || "line") : null;
  // Generate chart from raw results if user asked for plot/chart
  const chartDataFromResults = results && shouldShowChart ? formatChartDataFromResults(results) : null;
  // Priority: wage schedule chart > other charts > results-based chart
  const chartData = chartDataFromWageSchedule || chartDataFromCharts || chartDataFromResults;
  
  const isUser = role === "user";
  const hasStructuredAnswer = !isUser && !!answer_points && answer_points.length > 0;
  const hasSources = !isUser && !!sources && sources.length > 0;

  const [sourceLimit, setSourceLimit] = useState<number>(0);

  useEffect(() => {
    if (hasSources && sources) {
      setSourceLimit((prev) => {
        const defaultLimit = Math.min(5, Math.min(10, sources.length));
        if (prev === 0) {
          return defaultLimit;
        }
        return Math.min(prev, Math.min(10, sources.length));
      });
    } else {
      setSourceLimit(0);
    }
  }, [hasSources, sources?.length]);

  const visibleSources =
    hasSources && sources
      ? sources.slice(0, sourceLimit > 0 ? sourceLimit : Math.min(10, sources.length))
      : [];
  const maxSourceOptions = hasSources && sources ? Math.min(10, sources.length) : 0;

  // Custom components for markdown rendering
  const markdownComponents = {
    p: ({ children }: any) => (
      <p className="mb-2 last:mb-0">{children}</p>
    ),
    strong: ({ children }: any) => (
      <strong className="font-semibold text-foreground">{children}</strong>
    ),
    ul: ({ children }: any) => (
      <ul className="space-y-1 my-2">{children}</ul>
    ),
    li: ({ children }: any) => (
      <li className="flex items-start gap-2 min-w-0 max-w-full">
        <span className="text-primary mt-1 shrink-0">•</span>
        <span className="flex-1 break-words overflow-wrap-anywhere min-w-0">{children}</span>
      </li>
    ),
  };

  return (
    <div
      className={cn(
        "flex gap-3 px-4 py-6 animate-in fade-in slide-in-from-bottom-4 duration-500",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary to-secondary shadow-soft">
          <Bot className="h-5 w-5 text-primary-foreground" />
        </div>
      )}
      
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3 shadow-soft transition-all min-w-0 w-full overflow-hidden",
          isUser
            ? "bg-gradient-to-br from-primary to-secondary text-primary-foreground"
            : "bg-card border border-border"
        )}
      >
        {/* Show classification badge for assistant messages */}
        {!isUser && query_classification && (
          <div className="mb-2 flex items-center gap-2 min-w-0 max-w-full">
            <span
              className={cn(
                "rounded-full border px-2.5 py-1 text-xs font-medium tracking-wide break-words overflow-wrap-anywhere max-w-full",
                query_classification?.startsWith("Contract Knowledge")
                  ? "border-emerald-300/50 bg-emerald-500 text-white shadow-[0_0_0_1px_rgba(16,185,129,0.3)]"
                  : query_classification === "General Knowledge"
                  ? "border-slate-900/70 bg-slate-900 text-slate-100 shadow-[0_0_0_1px_rgba(15,23,42,0.4)]"
                  : "border-rose-300/50 bg-rose-500 text-white shadow-[0_0_0_1px_rgba(244,63,94,0.3)]"
              )}
            >
              Intent: {query_classification}
            </span>
          </div>
        )}
        
        <div className={cn(
          "text-sm leading-relaxed w-full max-w-full overflow-hidden",
          isUser ? "text-primary-foreground" : "text-foreground"
        )}>
          {isUser ? (
            <p className="whitespace-pre-wrap break-words overflow-wrap-anywhere">{content}</p>
          ) : hasStructuredAnswer ? (
            <div className="space-y-4 w-full max-w-full overflow-hidden">
              {opening && (
                <p className="text-sm font-semibold text-foreground leading-relaxed break-words overflow-wrap-anywhere">
                  {opening}
                </p>
              )}
              <ul className="mt-2 space-y-2 text-foreground w-full max-w-full">
                {answer_points?.map((point, idx) => (
                  <li key={idx} className="flex gap-2 w-full min-w-0 max-w-full">
                    <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-primary flex-shrink-0" />
                    <span className="flex-1 break-words overflow-wrap-anywhere min-w-0">{point}</span>
                  </li>
                ))}
              </ul>
              {disclaimer && (
                <div className="rounded-lg border border-border/60 bg-muted/20 px-3 py-2 text-sm text-muted-foreground break-words overflow-wrap-anywhere">
                  <strong className="font-semibold text-foreground">Disclaimer:</strong> {disclaimer}
                </div>
              )}
            </div>
          ) : (
            <ReactMarkdown components={markdownComponents}>
              {content}
            </ReactMarkdown>
          )}
        </div>

        {!hasStructuredAnswer && !isUser && disclaimer && (
          <div className="mt-4 rounded-lg border border-border/60 bg-muted/20 px-3 py-2 text-sm text-muted-foreground break-words overflow-wrap-anywhere">
            <strong className="font-semibold text-foreground">Disclaimer:</strong> {disclaimer}
          </div>
        )}

        {!isUser && hasSources && visibleSources.length > 0 && (
          <div className="mt-4 rounded-lg border border-border bg-background/50 p-4 w-full max-w-full overflow-hidden">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3 w-full min-w-0">
              <h4 className="text-sm font-semibold uppercase tracking-wide text-foreground shrink-0">Sources</h4>
              {maxSourceOptions > 1 && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span>Show</span>
                  <Select
                    value={String(sourceLimit || Math.min(3, maxSourceOptions))}
                    onValueChange={(value) => setSourceLimit(Number(value))}
                  >
                    <SelectTrigger className="h-8 w-[100px] border-border bg-card">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Array.from({ length: maxSourceOptions }, (_, idx) => idx + 1).map((count) => (
                        <SelectItem key={count} value={String(count)}>
                          {count}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <span>of {sources?.length}</span>
                </div>
              )}
            </div>
            <Accordion type="multiple" className="w-full max-w-full overflow-hidden">
              {visibleSources.map((entry, idx) => (
                <AccordionItem key={`${entry.source}-${idx}`} value={`source-${idx}`} className="w-full max-w-full">
                  <AccordionTrigger className="text-left text-sm font-medium text-foreground w-full max-w-full overflow-hidden [&>span]:w-full [&>span]:min-w-0">
                    <span className="flex flex-col items-start w-full min-w-0 max-w-full">
                      <span className="break-words overflow-wrap-anywhere w-full min-w-0 max-w-full">{`#${idx + 1} ${entry.source || "Unknown source"}`}</span>
                      <span className="text-xs text-muted-foreground break-words overflow-wrap-anywhere w-full min-w-0 max-w-full">
                        {entry.clause && entry.clause !== "intro"
                          ? `${entry.clause}${
                              entry.clause_heading ? ` – ${entry.clause_heading}` : ""
                            }`
                          : entry.section_heading || "Section unavailable"}
                        {entry.score !== undefined ? ` • Similarity ${entry.score.toFixed(4)}` : ""}
                      </span>
                    </span>
                  </AccordionTrigger>
                  <AccordionContent className="text-sm text-muted-foreground w-full max-w-full overflow-hidden">
                    {entry.excerpt ? (
                      <p className="whitespace-pre-wrap leading-relaxed break-words overflow-wrap-anywhere w-full max-w-full">{entry.excerpt}</p>
                    ) : (
                      <p>No excerpt available from this source.</p>
                    )}
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </div>
        )}
        
        {averageData && (
          <div className="mt-4 rounded-lg border border-border bg-background/50 p-4">
            <h4 className="mb-3 text-sm font-semibold uppercase tracking-wide text-foreground">Summary Highlights</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 px-3 font-semibold text-foreground">Metric</th>
                    <th className="text-right py-2 px-3 font-semibold text-foreground">Average</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(averageData).map(([key, value]) => (
                    <tr key={key} className="border-b border-border/50 last:border-0">
                      <td className="py-2 px-3 text-muted-foreground capitalize">{key}</td>
                      <td className="py-2 px-3 text-right font-medium text-foreground">
                        {typeof value === 'number' ? value.toFixed(2) : value}
                        {key.includes('duration') ? ' h' : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        
        {/* Display SQL query results as table */}
        {results && results.length > 0 && (response_type === "sql_query" || response_type === "wage_schedule_sql") && (
          <div className="mt-4 rounded-lg border border-border bg-background/50 p-4">
            <h4 className="text-sm font-semibold mb-3 text-foreground">
              Query Results {total_rows && total_rows > 10 ? `(Showing top 10 of ${total_rows} rows)` : `(${results.length} rows)`}
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    {Object.keys(results[0]).map((key) => (
                      <th 
                        key={key} 
                        className="text-left py-2 px-3 font-semibold text-foreground bg-muted/50"
                      >
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.slice(0, 10).map((row, rowIndex) => (
                    <tr 
                      key={rowIndex} 
                      className="border-b border-border/50 last:border-0 hover:bg-muted/30 transition-colors"
                    >
                      {Object.keys(results[0]).map((key) => (
                        <td 
                          key={key} 
                          className="py-2 px-3 text-muted-foreground"
                        >
                          {row[key] !== null && row[key] !== undefined 
                            ? (typeof row[key] === 'number' 
                                ? row[key].toLocaleString('en-US', { maximumFractionDigits: 2 })
                                : String(row[key]))
                            : '—'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Display charts when requested or available */}
        {chartData && chartData.length > 0 && (
          <div className="mt-4 rounded-lg border border-border bg-background/50 p-4">
            <h4 className="text-sm font-semibold mb-3 text-foreground">
              {chart_type === "bar" ? "Wage Rate Comparison" : "Wage Rate Trends"}
            </h4>
            <ResponsiveContainer width="100%" height={300}>
              {chart_type === "bar" ? (
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis 
                    tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                    labelStyle={{ color: 'hsl(var(--foreground))' }}
                  />
                  <Legend 
                    wrapperStyle={{ fontSize: '12px' }}
                  />
                  {Object.keys(chartData[0] || {})
                    .filter(key => key !== 'date')
                    .map((key, index) => {
                      const colors = [
                        'hsl(var(--primary))',
                        'hsl(207, 88%, 62%)',
                        'hsl(187, 70%, 55%)',
                        'hsl(166, 60%, 48%)',
                        'hsl(222, 70%, 65%)',
                        'hsl(200, 80%, 58%)'
                      ];
                      return (
                        <Bar 
                          key={key}
                          dataKey={key} 
                          fill={colors[index % colors.length]}
                          name={key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ')}
                        />
                      );
                    })}
                </BarChart>
              ) : (
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis 
                    tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                    labelStyle={{ color: 'hsl(var(--foreground))' }}
                  />
                  <Legend 
                    wrapperStyle={{ fontSize: '12px' }}
                    iconType="line"
                  />
                  {Object.keys(chartData[0] || {})
                    .filter(key => key !== 'date')
                    .map((key, index) => {
                      const colors = [
                        'hsl(var(--primary))',
                        'hsl(207, 88%, 62%)',
                        'hsl(187, 70%, 55%)',
                        'hsl(166, 60%, 48%)',
                        'hsl(222, 70%, 65%)',
                        'hsl(200, 80%, 58%)'
                      ];
                      return (
                        <Line 
                          key={key}
                          type="monotone" 
                          dataKey={key} 
                          stroke={colors[index % colors.length]}
                          strokeWidth={2}
                          name={key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ')}
                          dot={{ r: 4 }}
                          activeDot={{ r: 6 }}
                        />
                      );
                    })}
                </LineChart>
              )}
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
          <User className="h-5 w-5 text-muted-foreground" />
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
