import { Bot } from "lucide-react";
import { useState, useEffect, useMemo } from "react";

interface TypingIndicatorProps {
  query?: string;
  classification?: string;
}

const TypingIndicator = ({ query = "", classification }: TypingIndicatorProps) => {
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);

  // Determine query type from query text or classification
  const queryType = useMemo(() => {
    if (classification) {
      if (classification.includes("Wage Schedule")) return "wage_schedule";
      if (classification.includes("Contract Knowledge")) return "contract_knowledge";
      if (classification.includes("General Knowledge")) return "generic_knowledge";
      return "unknown";
    }
    
    // Fallback: infer from query text
    const lowerQuery = query.toLowerCase();
    const wageKeywords = ["wage", "salary", "pay rate", "plot", "tabulate", "list", "shift", "hourly rate", "fiscal year", "fy", "shift1", "shift2", "shift3"];
    const contractKeywords = [
      "contract", "clause", "article", "section", "agreement", 
      "vacation", "holiday", "holidays", "guarantee", "guarantees",
      "meetings", "meeting", "registered clerks", "guidelines", "guideline",
      "rules", "rule", "requirements", "requirement", "benefits", "benefit",
      "procedure", "procedures", "entitle", "entitled", "explain", "what are",
      "how many", "walking bosses", "foremen", "longshoremen", "clerks",
      "meal time", "meal period", "grievance", "arbitration"
    ];
    
    if (wageKeywords.some(keyword => lowerQuery.includes(keyword))) {
      return "wage_schedule";
    }
    if (contractKeywords.some(keyword => lowerQuery.includes(keyword))) {
      return "contract_knowledge";
    }
    return "generic";
  }, [query, classification]);

  // Generate messages based on query type
  const messages = useMemo(() => {
    const baseMessages = [
      "Classifying your query...",
    ];

    if (queryType === "wage_schedule") {
      return [
        "Classifying your query...",
        "Query classified as Wage Schedule · SQL",
        "Generating SQL query...",
        "Running query against wage database...",
        "Processing results...",
        "Summarizing the results...",
        "Almost there...",
      ];
    } else if (queryType === "contract_knowledge") {
      return [
        "Classifying your query...",
        "Query classified as Contract Knowledge",
        "Searching contracts...",
        "Analyzing relevant clauses...",
        "Synthesizing insights...",
        "Summarizing the results...",
        "Almost there...",
      ];
    } else if (queryType === "generic_knowledge") {
      return [
        "Classifying your query...",
        "Query classified as General Knowledge",
        "Looking at our general knowledge...",
        "Formulating response...",
        "Summarizing the results...",
        "Almost there...",
      ];
    } else {
      return [
        ...baseMessages,
        "Processing your request...",
        "Gathering insights...",
        "Summarizing the results...",
        "Almost there...",
      ];
    }
  }, [queryType]);

  useEffect(() => {
    // Start with first message
    setCurrentMessageIndex(0);
    
    // Rotate messages at different intervals
    // First message shows immediately, then progress through others
    const timers: NodeJS.Timeout[] = [];
    
    for (let i = 1; i < messages.length; i++) {
      // Increased delays: First message: 4s, Second: +5s (9s total), Others: +6s each
      // For "Almost there" (last message), add extra delay for contract_knowledge
      const isLastMessage = i === messages.length - 1;
      const isContractKnowledge = queryType === "contract_knowledge";
      
      let cumulativeDelay = i === 1 ? 4000 : i === 2 ? 9000 : 9000 + (i - 2) * 6000;
      
      // Add extra delay for "Almost there" in contract knowledge
      if (isLastMessage && isContractKnowledge) {
        cumulativeDelay += 8000; // Add 8 more seconds for contract knowledge
      }
      
      const timer = setTimeout(() => {
        setCurrentMessageIndex(i);
      }, cumulativeDelay);
      
      timers.push(timer);
    }

    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [messages, queryType]);

  return (
    <div className="flex gap-3 px-4 py-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary to-secondary shadow-soft animate-pulse">
        <Bot className="h-5 w-5 text-primary-foreground" />
      </div>
      
      <div className="max-w-[80%] rounded-2xl border border-border bg-card px-4 py-3 shadow-soft">
        <div className="flex items-center gap-3">
          <div className="flex gap-1">
            <div className="h-2 w-2 animate-bounce rounded-full bg-primary [animation-delay:-0.3s]" />
            <div className="h-2 w-2 animate-bounce rounded-full bg-primary [animation-delay:-0.15s]" />
            <div className="h-2 w-2 animate-bounce rounded-full bg-primary" />
          </div>
          <span className="text-sm text-muted-foreground transition-opacity duration-300">
            {messages[currentMessageIndex]}
          </span>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
