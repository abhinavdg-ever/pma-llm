import { Bot } from "lucide-react";
import { useState, useEffect } from "react";

const thinkingMessages = [
  "Thinking...",
  "Classifying your query...",
  "Planning next moves...",
  "Analyzing your sleep data...",
  "Thinking longer...",
  "Processing your request...",
  "Gathering insights...",
  "Almost there...",
];

const TypingIndicator = () => {
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);

  useEffect(() => {
    // Rotate messages every 6 seconds, but stop at the last message
    const interval = setInterval(() => {
      setCurrentMessageIndex((prevIndex) => {
        // If we're at the last message, stay there
        if (prevIndex >= thinkingMessages.length - 1) {
          return prevIndex;
        }
        // Otherwise, move to next message
        return prevIndex + 1;
      });
    }, 6000);

    return () => clearInterval(interval);
  }, []);

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
            {thinkingMessages[currentMessageIndex]}
          </span>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
