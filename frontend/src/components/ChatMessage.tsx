import { cn } from "@/lib/utils";
import { Bot, User } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import ReactMarkdown from 'react-markdown';

interface ChartData {
  chart_type: string;
  data: Record<string, any>;
  format: string;
}

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  response_type?: string;
  charts?: ChartData | null;
  results?: any[] | null;
  total_rows?: number;
  userQuery?: string; // Original user query to detect plot requests
  query_classification?: string; // SQL or Knowledge/General
}

const ChatMessage = ({ role, content, response_type, charts, results, total_rows, userQuery, query_classification }: ChatMessageProps) => {
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

  // Format chart data from SQL results (if results have date column)
  const formatChartDataFromResults = (results: any[]) => {
    if (!results || results.length === 0) return null;
    
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
  const shouldShowChart = userQuery && (
    userQuery.toLowerCase().includes('plot') || 
    userQuery.toLowerCase().includes('chart') || 
    userQuery.toLowerCase().includes('graph') ||
    userQuery.toLowerCase().includes('visualize') ||
    userQuery.toLowerCase().includes('trend')
  );
  
  const chartDataFromCharts = charts ? formatChartDataFromCharts(charts) : null;
  const chartDataFromResults = results && shouldShowChart ? formatChartDataFromResults(results) : null;
  const chartData = chartDataFromCharts || chartDataFromResults;
  
  const isUser = role === "user";

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
      <li className="flex items-start gap-2">
        <span className="text-primary mt-1">•</span>
        <span className="flex-1">{children}</span>
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
          "max-w-[85%] rounded-2xl px-4 py-3 shadow-soft transition-all",
          isUser
            ? "bg-gradient-to-br from-primary to-secondary text-primary-foreground"
            : "bg-card border border-border"
        )}
      >
        {/* Show classification badge for assistant messages */}
        {!isUser && query_classification && (
          <div className="mb-2 flex items-center gap-2">
            <span className={cn(
              "text-xs font-medium px-2.5 py-1 rounded-full border",
              query_classification === "Data Pull (SQL)"
                ? "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-800"
                : query_classification === "Knowledge"
                ? "bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-900/30 dark:text-purple-400 dark:border-purple-800"
                : "bg-green-50 text-green-700 border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-800"
            )}>
              Query classified as: {query_classification}
            </span>
          </div>
        )}
        
        <div className={cn(
          "text-sm leading-relaxed",
          isUser ? "text-primary-foreground" : "text-foreground"
        )}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{content}</p>
          ) : (
            <ReactMarkdown components={markdownComponents}>
              {content}
            </ReactMarkdown>
          )}
        </div>
        
        {averageData && (
          <div className="mt-4 p-4 bg-background/50 rounded-lg border border-border">
            <h4 className="text-sm font-semibold mb-3 text-foreground">Summary Statistics</h4>
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
        {results && results.length > 0 && response_type === "sql_query" && (
          <div className="mt-4 p-4 bg-background/50 rounded-lg border border-border">
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
          <div className="mt-4 p-4 bg-background/50 rounded-lg border border-border">
            <h4 className="text-sm font-semibold mb-3 text-foreground">Sleep Trends</h4>
            <ResponsiveContainer width="100%" height={300}>
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
                      'hsl(210, 100%, 60%)',
                      'hsl(280, 70%, 60%)',
                      'hsl(30, 100%, 60%)',
                      'hsl(160, 70%, 50%)',
                      'hsl(350, 80%, 60%)'
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
