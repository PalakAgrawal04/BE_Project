import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Mic, MicOff, Send } from "lucide-react";
import { toast } from "sonner";

interface QueryInputProps {
  onSubmit: (query: string) => void;
}

const QueryInput = ({ onSubmit }: QueryInputProps) => {
  const [query, setQuery] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const recognitionRef = useRef<any>(null);

  const handleVoiceToggle = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      toast.error("Voice input not supported in this browser");
      return;
    }

    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
      return;
    }

    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;

    recognitionRef.current.onstart = () => {
      setIsRecording(true);
      toast.info("Listening... Speak your query");
    };

    recognitionRef.current.onresult = (event: any) => {
      const transcript = Array.from(event.results)
        .map((result: any) => result[0])
        .map((result) => result.transcript)
        .join("");
      setQuery(transcript);
    };

    recognitionRef.current.onerror = (event: any) => {
      console.error("Speech recognition error:", event.error);
      toast.error("Error with voice input");
      setIsRecording(false);
    };

    recognitionRef.current.onend = () => {
      setIsRecording(false);
    };

    recognitionRef.current.start();
  };

  const handleSubmit = () => {
    if (!query.trim()) {
      toast.error("Please enter a query");
      return;
    }

    setIsProcessing(true);
    onSubmit(query);
    toast.success("Query submitted");
    
    setTimeout(() => {
      setQuery("");
      setIsProcessing(false);
    }, 500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="bg-card rounded-lg shadow-medium border border-border p-6">
      <div className="space-y-4">
        <div className="flex items-start gap-3">
          <div className="flex-1 relative">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter your query in plain English... (e.g., 'Show me all customers who made purchases last month')"
              className="min-h-[120px] resize-none pr-12"
              disabled={isProcessing}
            />
            {isRecording && (
              <div className="absolute top-3 right-3">
                <div className="flex items-center gap-2 text-destructive animate-pulse">
                  <span className="w-2 h-2 bg-destructive rounded-full"></span>
                  <span className="text-xs font-medium">Recording</span>
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <Button
            onClick={handleVoiceToggle}
            variant={isRecording ? "destructive" : "outline"}
            size="lg"
            className="gap-2"
          >
            {isRecording ? (
              <>
                <MicOff className="h-4 w-4" />
                Stop
              </>
            ) : (
              <>
                <Mic className="h-4 w-4" />
                Voice Input
              </>
            )}
          </Button>
          
          <Button
            onClick={handleSubmit}
            disabled={!query.trim() || isProcessing}
            size="lg"
            className="gap-2 flex-1 bg-gradient-primary"
          >
            <Send className="h-4 w-4" />
            {isProcessing ? "Processing..." : "Submit Query"}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default QueryInput;
