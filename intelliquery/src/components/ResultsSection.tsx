import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Database, FileText, Network } from "lucide-react";
import RelationshipGraph from "./RelationshipGraph";

interface ResultsSectionProps {
  tableData?: any[];
  documents?: any[];
  relationships?: any[];
}

const ResultsSection = ({ tableData, documents, relationships }: ResultsSectionProps) => {
  const [activeTab, setActiveTab] = useState("table");

  const hasResults = tableData || documents || relationships;

  if (!hasResults) {
    return null;
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold">Results</h2>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3 max-w-md">
          <TabsTrigger value="table" disabled={!tableData}>
            <Database className="h-4 w-4 mr-2" />
            Data
          </TabsTrigger>
          <TabsTrigger value="documents" disabled={!documents}>
            <FileText className="h-4 w-4 mr-2" />
            Documents
          </TabsTrigger>
          <TabsTrigger value="graph" disabled={!relationships}>
            <Network className="h-4 w-4 mr-2" />
            Relationships
          </TabsTrigger>
        </TabsList>

        <TabsContent value="table" className="mt-4">
          {tableData && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Database className="h-4 w-4" />
                  Database Results
                  <Badge variant="secondary" className="ml-auto">
                    {tableData.length} rows
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {Object.keys(tableData[0] || {}).map((key) => (
                          <TableHead key={key} className="font-semibold">
                            {key.charAt(0).toUpperCase() + key.slice(1)}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {tableData.map((row, idx) => (
                        <TableRow key={idx}>
                          {Object.values(row).map((value: any, cellIdx) => (
                            <TableCell key={cellIdx}>{value}</TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="documents" className="mt-4">
          {documents && (
            <div className="space-y-4">
              {documents.map((doc, idx) => (
                <Card key={idx} className="hover:shadow-medium transition-shadow">
                  <CardHeader>
                    <CardTitle className="text-base flex items-center gap-2">
                      <FileText className="h-4 w-4 text-accent" />
                      {doc.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <p className="text-sm text-muted-foreground">{doc.snippet}</p>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        {doc.source}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="graph" className="mt-4">
          {relationships && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Network className="h-4 w-4" />
                  Knowledge Graph
                  <Badge variant="secondary" className="ml-auto">
                    {relationships.length} relationships
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <RelationshipGraph relationships={relationships} />
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ResultsSection;
