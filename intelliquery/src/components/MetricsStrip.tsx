import { Clock, Activity, CheckCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface MetricsStripProps {
  responseTime?: number;
}

const MetricsStrip = ({ responseTime }: MetricsStripProps) => {
  return (
    <div className="bg-muted/50 border-b border-border">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-success" />
            <span className="text-muted-foreground">Status:</span>
            <Badge variant="outline" className="bg-success/10 text-success border-success/20">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              Ready
            </Badge>
          </div>
          
          {responseTime !== undefined && (
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-accent" />
              <span className="text-muted-foreground">Response Time:</span>
              <Badge variant="outline" className="bg-accent/10 text-accent border-accent/20 font-mono">
                {responseTime.toFixed(0)}ms
              </Badge>
            </div>
          )}

          <div className="ml-auto text-xs text-muted-foreground">
            All systems operational
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsStrip;
