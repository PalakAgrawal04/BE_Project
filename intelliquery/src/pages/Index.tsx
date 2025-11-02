import { useState } from "react";
import QueryInput from "@/components/QueryInput";
import IntentDisplay from "@/components/IntentDisplay";
import ResultsSection from "@/components/ResultsSection";
import QueryHistory from "@/components/QueryHistory";
import MetricsStrip from "@/components/MetricsStrip";
import HelpModal from "@/components/HelpModal";
import { HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface QueryResult {
  id: string;
  query: string;
  timestamp: Date;
  intent: "structured" | "unstructured" | "hybrid";
  sqlQuery?: string;
  documentSummary?: string;
  tableData?: any[];
  documents?: any[];
  relationships?: any[];
  responseTime: number;
}

const Index = () => {
  const [queries, setQueries] = useState<QueryResult[]>([]);
  const [currentResult, setCurrentResult] = useState<QueryResult | null>(null);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const handleQuerySubmit = (query: string) => {
    // Simulate query processing
    const responseTime = Math.random() * 2000 + 500;
    
    setTimeout(() => {
      const result: QueryResult = {
        id: Date.now().toString(),
        query,
        timestamp: new Date(),
        intent: Math.random() > 0.7 ? "hybrid" : Math.random() > 0.5 ? "structured" : "unstructured",
        responseTime,
        sqlQuery: Math.random() > 0.3 ? "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id WHERE orders.date > '2024-01-01'" : undefined,
        documentSummary: Math.random() > 0.3 ? "Found 5 relevant documents about customer behavior patterns" : undefined,
        tableData: Math.random() > 0.3 ? [
          { id: 1, name: "Alice Johnson", total: "$12,450", orders: 23 },
          { id: 2, name: "Bob Smith", total: "$8,920", orders: 15 },
          { id: 3, name: "Carol White", total: "$15,670", orders: 31 },
        ] : undefined,
        documents: Math.random() > 0.3 ? [
          { title: "Q4 Customer Analysis", snippet: "Analysis shows increased customer engagement...", source: "reports/2024-q4.pdf" },
          { title: "Market Trends 2024", snippet: "Key trends indicate a shift towards...", source: "research/trends.docx" },
        ] : undefined,
        relationships: Math.random() > 0.5 ? [
          { from: "Customer", to: "Order", type: "places" },
          { from: "Order", to: "Product", type: "contains" },
          { from: "Customer", to: "Support Ticket", type: "creates" },
        ] : undefined,
      };
      
      setCurrentResult(result);
      setQueries(prev => [result, ...prev]);
    }, responseTime);
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-gradient-primary border-b border-border shadow-soft">
          <div className="container mx-auto px-4 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-primary-foreground">IntelliQuery</h1>
                <p className="text-sm text-primary-foreground/80 mt-1">Unified Data Query Platform</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="text-primary-foreground hover:bg-primary-foreground/10"
                onClick={() => setIsHelpOpen(true)}
              >
                <HelpCircle className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </header>

        {/* Metrics Strip */}
        <MetricsStrip responseTime={currentResult?.responseTime} />

        {/* Main Content Area */}
        <main className="flex-1 container mx-auto px-4 py-6 space-y-6">
          {/* Query Input */}
          <QueryInput onSubmit={handleQuerySubmit} />

          {/* Intent Display */}
          {currentResult && (
            <IntentDisplay
              intent={currentResult.intent}
              sqlQuery={currentResult.sqlQuery}
              documentSummary={currentResult.documentSummary}
            />
          )}

          {/* Results Section */}
          {currentResult && (
            <ResultsSection
              tableData={currentResult.tableData}
              documents={currentResult.documents}
              relationships={currentResult.relationships}
            />
          )}
        </main>
      </div>

      {/* Query History Sidebar */}
      <QueryHistory
        queries={queries}
        isOpen={isHistoryOpen}
        onToggle={() => setIsHistoryOpen(!isHistoryOpen)}
        onSelectQuery={(query) => setCurrentResult(query)}
      />

      {/* Help Modal */}
      <HelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </div>
  );
};

export default Index;
