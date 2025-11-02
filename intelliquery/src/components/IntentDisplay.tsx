import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, FileText, GitBranch } from "lucide-react";

interface IntentDisplayProps {
  intent: "structured" | "unstructured" | "hybrid";
  sqlQuery?: string;
  documentSummary?: string;
}

const IntentDisplay = ({ intent, sqlQuery, documentSummary }: IntentDisplayProps) => {
  const getIntentConfig = () => {
    switch (intent) {
      case "structured":
        return {
          icon: Database,
          label: "Structured Query",
          color: "bg-success text-success-foreground",
          description: "Database query detected",
        };
      case "unstructured":
        return {
          icon: FileText,
          label: "Unstructured Search",
          color: "bg-accent text-accent-foreground",
          description: "Document search detected",
        };
      case "hybrid":
        return {
          icon: GitBranch,
          label: "Hybrid Query",
          color: "bg-warning text-warning-foreground",
          description: "Combined database & document search",
        };
    }
  };

  const config = getIntentConfig();
  const Icon = config.icon;

  return (
    <div className="space-y-4">
      {/* Intent Badge */}
      <div className="flex items-center gap-3">
        <Badge className={`${config.color} gap-2 px-3 py-1.5`}>
          <Icon className="h-4 w-4" />
          {config.label}
        </Badge>
        <span className="text-sm text-muted-foreground">{config.description}</span>
      </div>

      {/* SQL Query Display */}
      {sqlQuery && (
        <Card className="border-success/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Database className="h-4 w-4 text-success" />
              Generated SQL Query
            </CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="bg-muted p-4 rounded-md overflow-x-auto text-sm font-mono">
              <code className="text-foreground">{sqlQuery}</code>
            </pre>
          </CardContent>
        </Card>
      )}

      {/* Document Summary */}
      {documentSummary && (
        <Card className="border-accent/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="h-4 w-4 text-accent" />
              Document Search Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">{documentSummary}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default IntentDisplay;
