import { useState } from "react";
import QueryInput from "@/components/QueryInput";
import IntentDisplay from "@/components/IntentDisplay";
import ResultsSection from "@/components/ResultsSection";
import QueryHistory from "@/components/QueryHistory";
import MetricsStrip from "@/components/MetricsStrip";
import HelpModal from "@/components/HelpModal";
import { HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { postQuery, QueryResponse } from "@/lib/api";
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
      const resp = await postQuery(query, true);
      
      // Handle validation errors (defensive: backend may return unexpected shape)
      if (!resp.success) {
        const validation = resp.validation || null;
        const issuesArray = Array.isArray(validation?.issues)
          ? validation!.issues
          : Array.isArray(resp.issues)
          ? resp.issues
          : [];

        if (validation) {
          if (validation.suggested_rewrite) {
            const msg = issuesArray.length > 0
              ? `Query needs revision: ${issuesArray.join('. ')}\nSuggested: "${validation.suggested_rewrite}"`
              : `Suggested rewrite: "${validation.suggested_rewrite}"`;
            toast.error(msg);
            return;
          }

          const msg = issuesArray.length > 0
            ? `Invalid query: ${issuesArray.join('. ')}`
            : resp.error || resp.status || 'Invalid query';
          toast.error(msg);
          return;
        }

        // If we get here, resp.success is false but there's no validation object
        toast.error(resp.error || resp.status || 'Request failed');
        return;
      }

      const responseTime = Date.now() - start;
      
      // Map backend response into the UI's QueryResult shape
      const result: QueryResult = {
        id: String(Date.now()),
        query,
        timestamp: new Date(resp.timestamp || Date.now()),
        intent: resp.data?.intent_analysis?.type || "structured",
        responseTime,
        sqlQuery: resp.data?.generated_sql,
        documentSummary: resp.data?.similar_documents?.[0]?.metadata?.content,
        tableData: resp.data?.sql_results || [],
        documents: resp.data?.similar_documents || [],
        relationships: resp.data?.intent_analysis?.relationships || [],
      };

      setCurrentResult(result);
      setQueries((prev) => [result, ...prev]);
    } catch (err: any) {
      console.error("Query failed", err);
      const errorMsg = err.body?.validation?.suggested_rewrite
        ? `Query needs revision: ${err.body?.validation?.issues?.join(". ")}\nSuggested: "${err.body?.validation?.suggested_rewrite}"`
        : err?.message || "Failed to run query";
      toast.error(errorMsg);
    }
  };  return (
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
