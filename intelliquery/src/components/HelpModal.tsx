import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Database, FileText, GitBranch, Mic, Send } from "lucide-react";

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const HelpModal = ({ isOpen, onClose }: HelpModalProps) => {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl">Welcome to IntelliQuery</DialogTitle>
          <DialogDescription>
            Your unified platform for querying structured and unstructured data
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {/* Quick Start */}
          <section>
            <h3 className="text-lg font-semibold mb-3">Quick Start</h3>
            <ol className="space-y-2 text-sm text-muted-foreground">
              <li className="flex gap-2">
                <span className="font-medium text-foreground">1.</span>
                <span>Enter your query in plain English using the text box or voice input</span>
              </li>
              <li className="flex gap-2">
                <span className="font-medium text-foreground">2.</span>
                <span>Click Submit or press Enter to process your query</span>
              </li>
              <li className="flex gap-2">
                <span className="font-medium text-foreground">3.</span>
                <span>View results in the Data, Documents, or Relationships tabs</span>
              </li>
              <li className="flex gap-2">
                <span className="font-medium text-foreground">4.</span>
                <span>Access your query history from the sidebar on the right</span>
              </li>
            </ol>
          </section>

          {/* Query Types */}
          <section>
            <h3 className="text-lg font-semibold mb-3">Query Types</h3>
            <div className="space-y-3">
              <div className="flex items-start gap-3 p-3 rounded-lg bg-success/10 border border-success/20">
                <Database className="h-5 w-5 text-success mt-0.5" />
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-success">Structured Queries</span>
                    <Badge className="bg-success text-success-foreground text-xs">SQL</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Query databases with natural language. Example: "Show all orders from last month"
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 rounded-lg bg-accent/10 border border-accent/20">
                <FileText className="h-5 w-5 text-accent mt-0.5" />
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-accent">Unstructured Queries</span>
                    <Badge className="bg-accent text-accent-foreground text-xs">Documents</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Search through documents and text. Example: "Find reports about customer satisfaction"
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 rounded-lg bg-warning/10 border border-warning/20">
                <GitBranch className="h-5 w-5 text-warning mt-0.5" />
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-warning">Hybrid Queries</span>
                    <Badge className="bg-warning text-warning-foreground text-xs">Combined</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Combine database and document searches. Example: "Show sales data and related customer feedback"
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Features */}
          <section>
            <h3 className="text-lg font-semibold mb-3">Key Features</h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex items-start gap-2">
                <Mic className="h-4 w-4 text-primary mt-0.5" />
                <div>
                  <div className="font-medium">Voice Input</div>
                  <div className="text-muted-foreground text-xs">Speak your queries naturally</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Send className="h-4 w-4 text-primary mt-0.5" />
                <div>
                  <div className="font-medium">Instant Results</div>
                  <div className="text-muted-foreground text-xs">Fast query processing</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Database className="h-4 w-4 text-primary mt-0.5" />
                <div>
                  <div className="font-medium">SQL Generation</div>
                  <div className="text-muted-foreground text-xs">Automatic query creation</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <GitBranch className="h-4 w-4 text-primary mt-0.5" />
                <div>
                  <div className="font-medium">Relationship Graphs</div>
                  <div className="text-muted-foreground text-xs">Visualize data connections</div>
                </div>
              </div>
            </div>
          </section>

          {/* Example Queries */}
          <section>
            <h3 className="text-lg font-semibold mb-3">Example Queries</h3>
            <div className="space-y-2 text-sm">
              <div className="p-2 rounded bg-muted font-mono">
                "Show me all customers who made purchases last month"
              </div>
              <div className="p-2 rounded bg-muted font-mono">
                "Find documents mentioning product feedback"
              </div>
              <div className="p-2 rounded bg-muted font-mono">
                "What are the top performing products and their reviews?"
              </div>
            </div>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default HelpModal;
