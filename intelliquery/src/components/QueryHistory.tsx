import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { ChevronLeft, ChevronRight, Clock, Database, FileText, GitBranch } from "lucide-react";
import { QueryResult } from "@/pages/Index";
import { formatDistanceToNow } from "date-fns";

interface QueryHistoryProps {
  queries: QueryResult[];
  isOpen: boolean;
  onToggle: () => void;
  onSelectQuery: (query: QueryResult) => void;
}

const QueryHistory = ({ queries, isOpen, onToggle, onSelectQuery }: QueryHistoryProps) => {
  const getIntentIcon = (intent: QueryResult["intent"]) => {
    switch (intent) {
      case "structured":
        return <Database className="h-3 w-3" />;
      case "unstructured":
        return <FileText className="h-3 w-3" />;
      case "hybrid":
        return <GitBranch className="h-3 w-3" />;
    }
  };

  const getIntentColor = (intent: QueryResult["intent"]) => {
    switch (intent) {
      case "structured":
        return "bg-success text-success-foreground";
      case "unstructured":
        return "bg-accent text-accent-foreground";
      case "hybrid":
        return "bg-warning text-warning-foreground";
    }
  };

  return (
    <>
      {/* Toggle Button */}
      <Button
        onClick={onToggle}
        variant="outline"
        size="icon"
        className="fixed right-4 top-1/2 -translate-y-1/2 z-50 shadow-medium"
      >
        {isOpen ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
      </Button>

      {/* Sidebar */}
      <aside
        className={`fixed right-0 top-0 h-full bg-card border-l border-border shadow-strong transition-transform duration-300 z-40 ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
        style={{ width: "360px" }}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="p-4 border-b border-border">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Clock className="h-5 w-5 text-primary" />
              Query History
            </h2>
            <p className="text-xs text-muted-foreground mt-1">
              {queries.length} {queries.length === 1 ? "query" : "queries"}
            </p>
          </div>

          {/* History List */}
          <ScrollArea className="flex-1">
            <div className="p-4 space-y-3">
              {queries.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No queries yet</p>
                </div>
              ) : (
                queries.map((query) => (
                  <button
                    key={query.id}
                    onClick={() => onSelectQuery(query)}
                    className="w-full text-left p-3 rounded-lg border border-border hover:bg-muted/50 hover:border-primary/50 transition-all group"
                  >
                    <div className="flex items-start gap-2 mb-2">
                      <Badge className={`${getIntentColor(query.intent)} gap-1 text-xs px-2 py-0.5`}>
                        {getIntentIcon(query.intent)}
                        {query.intent}
                      </Badge>
                      <span className="text-xs text-muted-foreground ml-auto">
                        {formatDistanceToNow(query.timestamp, { addSuffix: true })}
                      </span>
                    </div>
                    <p className="text-sm line-clamp-2 group-hover:text-primary transition-colors">
                      {query.query}
                    </p>
                    <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {query.responseTime.toFixed(0)}ms
                    </div>
                  </button>
                ))
              )}
            </div>
          </ScrollArea>
        </div>
      </aside>
    </>
  );
};

export default QueryHistory;
