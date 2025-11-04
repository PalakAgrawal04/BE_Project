import { useState } from "react";
import QueryInput from "@/components/QueryInput";
import IntentDisplay from "@/components/IntentDisplay";
import ResultsSection from "@/components/ResultsSection";
import QueryHistory from "@/components/QueryHistory";
import MetricsStrip from "@/components/MetricsStrip";
import HelpModal from "@/components/HelpModal";
import { HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { postIntent } from "@/lib/api";
import { toast } from "sonner";

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

  const handleQuerySubmit = async (query: string) => {
    const start = Date.now();
    try {
      const resp = await postIntent(query, true);

      const responseTime = Date.now() - start;

      // Map backend response into the UI's QueryResult shape as best-effort
      const result: QueryResult = {
        id: String(Date.now()),
        query,
        timestamp: new Date(resp.timestamp ? resp.timestamp : Date.now()),
        intent: (resp.intent_analysis?.type as any) || (resp.metadata?.inferred_intent as any) || "unstructured",
        responseTime,
        sqlQuery: resp.intent_analysis?.sql || resp.metadata?.sql_query,
        documentSummary: resp.intent_analysis?.document_summary || resp.metadata?.document_summary,
        tableData: resp.metadata?.tableData || undefined,
        documents: resp.metadata?.documents || undefined,
        relationships: resp.metadata?.relationships || undefined,
      };

      setCurrentResult(result);
      setQueries((prev) => [result, ...prev]);
    } catch (err: any) {
      console.error("Query failed", err);
      toast.error(err?.message || "Failed to run query");
    }
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
