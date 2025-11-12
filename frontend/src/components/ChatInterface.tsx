import { useState, useRef, useEffect } from "react";
import { useToast } from "@/components/ui/use-toast";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import TypingIndicator from "./TypingIndicator";
import { Anchor, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";

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
}

interface Message {
  role: "user" | "assistant";
  content: string;
  response_type?: string;
  charts?: ChartData | null;
  debug?: any;
  results?: any[] | null;
  total_rows?: number;
  userQuery?: string; // Store original user query for context
  query_classification?: string; // SQL or Knowledge/General
  answer_points?: string[];
  disclaimer?: string | null;
  sources?: SourceEntry[];
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleClearChat = () => {
    setMessages([]);
    toast({
      title: "Chat cleared",
      description: "All messages have been removed.",
    });
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = { role: "user", content, userQuery: content };
    setMessages((prev) => [...prev, userMessage]);
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
          debug: data.debug,
          results: data.results || null,
          total_rows: data.total_rows || 0,
          userQuery: content, // Pass the original user query for chart detection
          query_classification: data.query_classification || null,
          answer_points: data.answer_points || [],
          disclaimer: data.disclaimer ?? null,
          sources: data.sources || []
        },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      toast({
        title: "Error",
        description: "Failed to get response from the contract engine. Please confirm the backend API is running on http://localhost:8000",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-gradient-to-b from-[#0f172a] via-[#0b1b33] to-[#0f172a] text-foreground">
      {/* Header */}
      <div className="border-b border-border bg-card/60 backdrop-blur-sm px-4 py-4 shadow-soft">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-primary to-secondary shadow-medium">
              <Anchor className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-foreground">Contract Insights Engine</h1>
              <p className="text-sm text-muted-foreground">
                Navigate longshore agreements with maritime-grade intelligence
              </p>
            </div>
          </div>
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

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl py-6">
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
                </ul>
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
              results={message.results}
              total_rows={message.total_rows}
              userQuery={message.userQuery}
            query_classification={message.query_classification}
            answer_points={message.answer_points}
            disclaimer={message.disclaimer}
            sources={message.sources}
            />
          ))}
          
          {isLoading && <TypingIndicator />}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <ChatInput onSend={handleSendMessage} disabled={isLoading} />
    </div>
  );
};

export default ChatInterface;
