export interface IntentResponse {
  success: boolean;
  query?: string;
  intent_analysis?: any;
  metadata?: any;
  validation?: any;
  timestamp?: string;
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
