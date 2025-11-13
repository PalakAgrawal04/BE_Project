export interface ValidationResult {
  is_coherent: boolean;
  is_valid: boolean;
  issues: string[];
  suggested_rewrite?: string;
  final_decision: 'valid' | 'needs_clarification' | 'invalid';
}

export interface QueryResponse {
  success: boolean;
  data?: {
    intent_analysis?: {
      type: "structured" | "unstructured" | "hybrid";
      relationships?: any[];
    };
    generated_sql?: string;
    sql_results?: any[];
    similar_documents?: any[];
  };
  execution_time?: number;
  query_type?: string;
  timestamp?: string;
  error?: string;
  status?: string;
  validation?: ValidationResult;
  suggested_rewrite?: string;
  issues?: string[];
}

export interface IntentResponse {
  success: boolean;
  query?: string;
  intent_analysis?: any;
  metadata?: any;
  validation?: any;
  timestamp?: string;
}

export async function postQuery(query: string, include_similar = true): Promise<QueryResponse> {
  // Use the correct port from environment or default to 5000
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000';
  const res = await fetch(`${apiUrl}/api/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      query, 
      include_similar_documents: include_similar 
    }),
  });

  const json = await res.json().catch(() => ({ success: false, error: 'invalid_json' }));

  // Handle validation errors separately from other errors
  if (!res.ok) {
    if (res.status === 400 && json?.status === 'invalid_query') {
      return json as QueryResponse;  // Return validation results
    }
    const err: any = new Error(json?.message || json?.error || 'Request failed');
    err.status = res.status;
    err.body = json;
    throw err;
  }

  return json as QueryResponse;
}

export async function postIntent(query: string, include_metadata = true): Promise<IntentResponse> {
  const res = await fetch('/api/intent', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, include_metadata }),
  });

  const json = await res.json().catch(() => ({ success: false, error: 'invalid_json' }));

  if (!res.ok) {
    const err: any = new Error(json?.message || 'Request failed');
    err.status = res.status;
    err.body = json;
    throw err;
  }

  return json as IntentResponse;
}

export async function postValidate(query: string): Promise<any> {
  const res = await fetch('/api/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  const json = await res.json();
  if (!res.ok) throw new Error(json?.message || 'Validation failed');
  return json;
}
