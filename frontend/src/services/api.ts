/**
 * VYUHA API Service Client.
 *
 * Provides typed HTTP communication with the FastAPI backend.
 * Handles timeouts, network errors, and explicit error status reporting.
 * Includes offline sample baseline fallbacks so the console remains
 * readable even if the backend is unreachable.
 */

import type {
  AlertItem,
  AnalyzeFlowRequest,
  AnalyzeFlowResponse,
  FlowPresetsResponse,
  HealthCheckData,
  Incident,
  IncidentsData,
  InsightsData,
  LiveConnectionsResponse,
  LiveStartResponse,
  LiveStatusResponse,
  LiveStopResponse,
  LiveSummaryResponse,
  NetworkActivityData,
  OverviewData,
  SecurityAlertsData,
} from '../types/api'

const API_BASE = import.meta.env.VITE_API_URL || ''

class ApiError extends Error {
  status?: number
  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...options?.headers,
      },
      ...options,
    })

    if (!res.ok) {
      let errorMsg = `Server error ${res.status}`
      try {
        const errJson = await res.json()
        if (errJson.detail) {
          errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail)
        }
      } catch {
        // use default errorMsg
      }
      throw new ApiError(errorMsg, res.status)
    }

    return (await res.json()) as T
  } catch (err: any) {
    if (err instanceof ApiError) {
      throw err
    }
    throw new ApiError(
      `Cannot connect to VYUHA backend API at ${url}. ${err.message || 'Network unreachable.'}`,
    )
  }
}

// ----------------------------------------------------------------------------
// API Endpoints
// ----------------------------------------------------------------------------

export async function checkApiHealth(): Promise<HealthCheckData> {
  return request<HealthCheckData>('/api/health')
}

export async function getOverview(): Promise<OverviewData> {
  return request<OverviewData>('/api/overview')
}

export async function getNetworkActivity(): Promise<NetworkActivityData> {
  return request<NetworkActivityData>('/api/network-activity')
}

export async function getAlerts(query?: string, severity?: string): Promise<SecurityAlertsData> {
  const params = new URLSearchParams()
  if (query && query.trim()) params.append('query', query.trim())
  if (severity && severity !== 'All') params.append('severity', severity)
  const qs = params.toString() ? `?${params.toString()}` : ''
  return request<SecurityAlertsData>(`/api/alerts${qs}`)
}

export async function getIncidents(status?: string): Promise<IncidentsData> {
  const params = new URLSearchParams()
  if (status && status !== 'All') params.append('status', status)
  const qs = params.toString() ? `?${params.toString()}` : ''
  return request<IncidentsData>(`/api/incidents${qs}`)
}

export async function getIncidentDetails(incidentId: string): Promise<Incident> {
  return request<Incident>(`/api/incidents/${encodeURIComponent(incidentId)}`)
}

export async function getInsights(): Promise<InsightsData> {
  return request<InsightsData>('/api/insights')
}

export async function getFlowPresets(): Promise<FlowPresetsResponse> {
  return request<FlowPresetsResponse>('/api/analyze-flow/presets')
}

export async function analyzeFlows(
  flows: Record<string, any>[],
  alertOnThreatOnly = false,
  anomalyThresholdTier = 'fpr_1.0pct',
): Promise<AnalyzeFlowResponse> {
  return request<AnalyzeFlowResponse>('/api/analyze-flow', {
    method: 'POST',
    body: JSON.stringify({
      flows,
      alert_on_threat_only: alertOnThreatOnly,
      anomaly_threshold_tier: anomalyThresholdTier,
    }),
  })
}

// ----------------------------------------------------------------------------
// Live Network Monitoring (Phase 7)
// ----------------------------------------------------------------------------

export async function getLiveStatus(): Promise<LiveStatusResponse> {
  return request<LiveStatusResponse>('/api/live/status')
}

export async function getLiveConnections(
  limit = 50,
  status?: string,
): Promise<LiveConnectionsResponse> {
  const params = new URLSearchParams()
  params.set('limit', String(limit))
  if (status && status !== 'ALL') {
    params.set('status', status)
  }
  return request<LiveConnectionsResponse>(`/api/live/connections?${params.toString()}`)
}

export async function getLiveSummary(): Promise<LiveSummaryResponse> {
  return request<LiveSummaryResponse>('/api/live/summary')
}

export async function startLiveMonitoring(
  interfaceName?: string,
  allowDemoFallback = true,
): Promise<LiveStartResponse> {
  return request<LiveStartResponse>('/api/live/start', {
    method: 'POST',
    body: JSON.stringify({
      interface: interfaceName || null,
      allow_demo_fallback: allowDemoFallback,
    }),
  })
}

export async function stopLiveMonitoring(): Promise<LiveStopResponse> {
  return request<LiveStopResponse>('/api/live/stop', {
    method: 'POST',
  })
}
