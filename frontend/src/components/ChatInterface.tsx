import { useState, useRef, useEffect } from "react";
import { useToast } from "@/components/ui/use-toast";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import TypingIndicator from "./TypingIndicator";
import { Anchor, Trash2, Lightbulb, Sparkles, ChevronRight, HelpCircle, Copy, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

// Configurable user ID - can be set via environment variable
const DEFAULT_USER_ID = import.meta.env.VITE_DEFAULT_USER_ID || "12";

// Use proxy in dev mode, or relative URL in production (nginx handles proxying)
const getApiUrl = () => {
  if (import.meta.env.DEV) {
    return "/api"; // Use proxy from vite.config.ts
  }
  // In production, use relative URL so nginx can proxy the request
  // Nginx is configured to proxy /api/* to FastAPI on port 8001
  return import.meta.env.VITE_API_URL || "/api";
};
const API_URL = getApiUrl();

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

interface Message {
  role: "user" | "assistant";
  content: string;
  response_type?: string;
  charts?: ChartData | null;
  chart?: any;
  chart_type?: string;
  debug?: any;
  results?: any[] | null;
  total_rows?: number;
  userQuery?: string; // Store original user query for context
  query_classification?: string; // SQL or Knowledge/General
  answer_points?: string[];
  disclaimer?: string | null;
  sources?: SourceEntry[];
  opening?: string | null;
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentQuery, setCurrentQuery] = useState<string>("");
  const [usedSampleQueries, setUsedSampleQueries] = useState<Set<string>>(new Set());
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const sampleQueries = [
    {
      category: "Wage Schedule",
      queries: [
        "Plot the latest salary for walking bosses for current year",
        "Show wage rates for Skill 2 longshoremen with <1000 hours of experience from FY25 to FY27",
        "Tabulate hourly rates for clerks in fiscal year 2025",
      ]
    },
    {
      category: "Contract Knowledge",
      queries: [
        "Explain the guidelines for registered clerks meetings",
        "How many vacation days do longshoremen get?",
        "What are the guarantee rules for walking bosses?",
      ]
    }
  ];

  const tips = [
    {
      title: "For Wage Queries",
      items: [
        "As per contract: Wage data is sourced from the official wage schedule database",
        "Provide Employee Type, Skill Type, and Experience Hours for accurate results",
        "Use specific fiscal year ranges (e.g., FY25 to FY27) for better filtering",
        "Include shift information (Shift 1, Shift 2, Shift 3) to get specific shift rates",
      ]
    },
    {
      title: "For Contract Questions",
      items: [
        "Be specific about the document type (longshore, clerks, walking bosses)",
        "Mention section or article numbers if you know them",
        "Ask about specific topics (vacations, holidays, guarantees, etc.)",
      ]
    },
    {
      title: "General Tips",
      items: [
        "Use natural language - the system understands conversational queries",
        "You can ask for plots, tables, or summaries",
        "Questions about percentages and trends are supported",
      ]
    }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleClearChat = () => {
    setMessages([]);
    setUsedSampleQueries(new Set());
    toast({
      title: "Chat cleared",
      description: "All messages have been removed.",
    });
  };

  const handleSampleQueryClick = (query: string) => {
    if (isLoading) {
      return; // Prevent submission if already loading
    }
    setUsedSampleQueries(prev => new Set(prev).add(query));
    setIsHelpOpen(false);
    handleSendMessage(query);
  };

  const handleCopyQuery = async (query: string) => {
    try {
      await navigator.clipboard.writeText(query);
      toast({
        title: "Copied!",
        description: "Query text copied to clipboard",
      });
    } catch (error) {
      console.error("Failed to copy:", error);
      toast({
        title: "Failed to copy",
        description: "Could not copy to clipboard",
        variant: "destructive",
      });
    }
  };

  const handleResubmitQuery = (query: string) => {
    if (isLoading) {
      return; // Prevent submission if already loading
    }
    setIsHelpOpen(false);
    handleSendMessage(query);
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = { role: "user", content, userQuery: content };
    setMessages((prev) => [...prev, userMessage]);
    setCurrentQuery(content);
    setIsLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/query`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ 
            user_id: DEFAULT_USER_ID,
            query: content 
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to get response from contract engine");
      }

      // Parse the JSON response from the API
      const data = await response.json();

      // Add the complete assistant message with full response data
      setMessages((prev) => [
        ...prev,
        { 
          role: "assistant", 
          content: data.content || 'No response received',
          response_type: data.response_type,
          charts: data.charts,
          chart: data.chart, // Wage schedule chart
          chart_type: data.chart_type, // Wage schedule chart type
          debug: data.debug,
          results: data.results || null,
          total_rows: data.total_rows || 0,
          userQuery: content, // Pass the original user query for chart detection
          query_classification: data.query_classification || null,
          answer_points: data.answer_points || [],
          disclaimer: data.disclaimer ?? null,
          sources: data.sources || [],
          opening: data.opening ?? null,
        },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      toast({
        title: "Error",
        description: "Can't process your request currently. Please retry in a moment.",
        variant: "destructive",
      });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Can't process your request currently. Please retry in a moment.",
          response_type: "error",
          userQuery: content,
          opening: null,
        },
      ]);
    } finally {
      setIsLoading(false);
      setCurrentQuery("");
    }
  };

  return (
    <div className="flex h-screen flex-col bg-gradient-to-b from-[#d9f0ff] via-[#bfe3ff] to-[#ecf8ff] text-foreground">
      {/* Header */}
      <div className="border-b border-border bg-card/60 backdrop-blur-sm px-4 py-4 shadow-soft">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-primary to-secondary shadow-medium">
              <Anchor className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-foreground">Contracts Copilot</h1>
              <p className="text-sm text-muted-foreground">
                Navigate longshore agreements with maritime-grade intelligence
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Dialog open={isHelpOpen} onOpenChange={setIsHelpOpen}>
              <DialogTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="shrink-0 flex items-center gap-2"
                  title="Help & Tips"
                >
                  <HelpCircle className="h-4 w-4" />
                  <span className="hidden sm:inline">Help</span>
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto overflow-x-hidden">
                <DialogHeader>
                  <DialogTitle className="flex items-center gap-2">
                    <Lightbulb className="h-5 w-5 text-primary" />
                    Tips & Sample Queries
                  </DialogTitle>
                  <DialogDescription>
                    Get tips on how to query the system and try out sample queries
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-6 mt-4 overflow-x-hidden break-words">
                  {/* Tips Section */}
                  <div>
                    <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-primary" />
                      Quick Tips
                    </h3>
                    <Accordion type="multiple" className="w-full">
                      {tips.map((tip, tipIndex) => (
                        <AccordionItem key={tipIndex} value={`tip-${tipIndex}`} className="border-border">
                          <AccordionTrigger className="text-sm font-medium text-foreground py-2">
                            {tip.title}
                          </AccordionTrigger>
                          <AccordionContent>
                            <ul className="space-y-2 text-sm text-muted-foreground pl-4">
                              {tip.items.map((item, itemIndex) => (
                                <li key={itemIndex} className="list-disc">{item}</li>
                              ))}
                            </ul>
                          </AccordionContent>
                        </AccordionItem>
                      ))}
                    </Accordion>
                  </div>

                  {/* Sample Queries Section */}
                  <div>
                    <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-primary" />
                      Sample Queries
                    </h3>
                    <div className="space-y-4">
                      {sampleQueries.map((category, catIndex) => (
                        <div key={catIndex}>
                          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                            {category.category}
                          </h4>
                          <div className="space-y-2">
                            {category.queries.map((query, queryIndex) => {
                              const isUsed = usedSampleQueries.has(query);
                              return (
                                <div key={queryIndex} className={`rounded-md border transition-colors ${
                                  isUsed 
                                    ? 'border-border/50 bg-muted/30' 
                                    : 'border-border bg-background hover:border-primary/50'
                                }`}>
                                  {isUsed ? (
                                    <Collapsible defaultOpen={false}>
                                      <CollapsibleTrigger asChild>
                                        <Button
                                          variant="ghost"
                                          className="w-full justify-between text-left h-auto py-3 px-3 text-sm font-normal text-muted-foreground break-words overflow-wrap-anywhere"
                                        >
                                          <div className="flex-1 text-left min-w-0 break-words overflow-wrap-anywhere">
                                            {query}
                                            <span className="ml-2 text-xs text-muted-foreground italic">
                                              (Used)
                                            </span>
                                          </div>
                                          <ChevronRight className="h-4 w-4 shrink-0 transition-transform duration-200 data-[state=open]:rotate-90 ml-2" />
                                        </Button>
                                      </CollapsibleTrigger>
                                      <CollapsibleContent className="px-3 pb-3">
                                        <div className="flex flex-col gap-2">
                                          <p className="text-xs text-muted-foreground mb-1">
                                            This query has been submitted.
                                          </p>
                                          <div className="flex gap-2">
                                            <Button
                                              variant="outline"
                                              size="sm"
                                              className="flex-1 h-8 text-xs"
                                              onClick={() => handleResubmitQuery(query)}
                                              disabled={isLoading}
                                            >
                                              <RotateCw className="h-3 w-3 mr-1.5" />
                                              Resubmit
                                            </Button>
                                            <Button
                                              variant="outline"
                                              size="sm"
                                              className="flex-1 h-8 text-xs"
                                              onClick={() => handleCopyQuery(query)}
                                              disabled={isLoading}
                                            >
                                              <Copy className="h-3 w-3 mr-1.5" />
                                              Copy
                                            </Button>
                                          </div>
                                        </div>
                                      </CollapsibleContent>
                                    </Collapsible>
                                  ) : (
                                    <Button
                                      variant="ghost"
                                      className={`w-full justify-start text-left h-auto py-3 px-3 text-sm font-normal break-words overflow-wrap-anywhere ${
                                        isLoading 
                                          ? 'text-muted-foreground/50 cursor-not-allowed opacity-60' 
                                          : 'text-foreground hover:text-primary'
                                      }`}
                                      onClick={() => handleSampleQueryClick(query)}
                                      disabled={isLoading}
                                    >
                                      <span className="break-words overflow-wrap-anywhere">{query}</span>
                                    </Button>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
            <Button
              variant={messages.length > 0 ? "default" : "outline"}
              size="sm"
              onClick={handleClearChat}
              className="shrink-0 flex items-center gap-2"
              title="Clear chat"
              disabled={messages.length === 0}
            >
              <Trash2 className="h-4 w-4" />
              {messages.length > 0 && <span>Clear Chat</span>}
            </Button>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl py-6 px-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-4 px-4 py-12 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-primary to-secondary shadow-medium">
                <Anchor className="h-8 w-8 text-primary-foreground" />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-foreground">Welcome aboard</h2>
                <p className="mt-2 text-muted-foreground">
                  Ask anything about ILWU/PMA contract language, interpretations, and clauses. Training set includes:
                </p>
                <ul className="mt-3 space-y-1 list-disc list-inside text-sm text-muted-foreground/90">
                  <li>Pacific Coast Longshore Contract Document (2022-2028)</li>
                  <li>Pacific Coast Walking Bosses and Foremen's Agreement (2022-2028)</li>
                  <li>Pacific Coast Clerks Contract Document (2022-2028)</li>
                  <li>Shift Wise Hourly Wage Rate Database By Employee Type (2022-2028)</li>
                </ul>
                <p className="mt-4 text-sm text-muted-foreground/80 italic">
                  Look at help section for sample queries and tips
                </p>
              </div>
            </div>
          )}
          
          {messages.map((message, index) => (
            <ChatMessage 
              key={index} 
              role={message.role} 
              content={message.content}
              response_type={message.response_type}
              charts={message.charts}
              chart={message.chart}
              chart_type={message.chart_type}
              results={message.results}
              total_rows={message.total_rows}
              userQuery={message.userQuery}
              query_classification={message.query_classification}
            answer_points={message.answer_points}
            disclaimer={message.disclaimer}
              sources={message.sources}
              opening={message.opening}
            />
          ))}
          
          {isLoading && <TypingIndicator query={currentQuery} />}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <ChatInput onSend={handleSendMessage} disabled={isLoading} />
    </div>
  );
};

export default ChatInterface;
