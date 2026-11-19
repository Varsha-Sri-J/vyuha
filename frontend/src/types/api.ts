/**
 * VYUHA TypeScript Data Contract & API Types.
 *
 * Reflects the FastAPI backend schemas while preserving the
 * exact visual tokens, severities, and status labels required by
 * the Manus editorial UI design system.
 */

export type Page = 'overview' | 'activity' | 'alerts' | 'incidents' | 'details' | 'insights'

export type Severity = 'High' | 'Medium' | 'Low'

export type IncidentStatus = 'Review' | 'Monitoring' | 'Closed candidate'

export type Tone = 'sage' | 'terracotta' | 'yellow' | 'lavender'

export type Confidence = 'Low' | 'Moderate' | 'Strong'

export interface AlertItem {
  id: string
  title: string
  category: string
  source: string
  severity: Severity
  summary: string
  time: string
  detail: string
  evidence: string
  whyFlagged: string
  technical: string
  threat_status?: string
  flow_id?: string
  destination_port?: number
  confidence?: number
  anomaly_score?: number
  is_attack?: boolean
}

export interface TimelineEvent {
  time: string
  title: string
  copy: string
  tone: Tone
}

export interface IncidentEvidenceItem {
  icon: string
  title: string
  copy: string
  tone: Tone
}

export interface Incident {
  id: string
  title: string
  summary: string
  severity: Severity
  alerts: number
  categories: string[]
  status: IncidentStatus
  updated: string
  engines: string[]
  relationship: string
  timeline: TimelineEvent[]
  nextSteps: string[]
  confidence: Confidence
  evidence: IncidentEvidenceItem[]
  missing_evidence?: string[]
  alert_ids?: string[]
}

export interface TrafficWindow {
  name: string
  incoming: number
  outgoing: number
  unusual: number
}

export interface WeekDayData {
  name: string
  normal: number
  unusual: number
}

export interface CategoryDataPoint {
  name: string
  value: number
  color: string
}

export interface EngineMetric {
  name: string
  matched: number
  reviewed: number
}

export interface MomentItem {
  time: string
  title: string
  copy: string
}

export interface FlowTotals {
  incoming: number
  outgoing: number
  suspicious: number
  total_observed: number
}

export interface AttentionItemData {
  title: string
  copy: string
  type?: string
  tone?: Tone
  target?: Page
  icon?: string
}

export interface OverviewData {
  posture: string
  posture_detail: string
  posture_summary: string
  flow_totals: FlowTotals
  categories_count: number
  open_incidents_count: number
  traffic_data: TrafficWindow[]
  recent_alerts: AlertItem[]
  attention_items: AttentionItemData[]
  data_source: string
  is_live_telemetry: boolean
}

export interface NetworkActivityData {
  flow_totals: FlowTotals
  traffic_data: TrafficWindow[]
  week_data: WeekDayData[]
  moments: MomentItem[]
  data_source: string
  is_live_telemetry: boolean
  disclaimer: string
}

export interface SecurityAlertsData {
  alerts: AlertItem[]
  total_count: number
  suspicious_flow_count: number
  data_source: string
  is_live_telemetry: boolean
}

export interface IncidentsData {
  incidents: Incident[]
  total_count: number
  open_count: number
  data_source: string
  is_live_telemetry: boolean
}

export interface InsightsData {
  category_data: CategoryDataPoint[]
  week_data: WeekDayData[]
  engine_data: EngineMetric[]
  benchmark_metrics: Record<string, any>
  data_source: string
  is_live_telemetry: boolean
}

export interface HealthCheckData {
  status: string
  service: string
  version: string
  data_provenance: string
  is_live_telemetry: boolean
  models_loaded?: boolean
}

// ----------------------------------------------------------------------------
// Phase 6: Machine Learning Inference Types
// ----------------------------------------------------------------------------

export interface BinaryEvaluationDetail {
  is_attack: boolean
  attack_probability: number
  confidence: number
  decision_threshold: number
}

export interface MulticlassEvaluationDetail {
  predicted_category: string
  category_probabilities: Record<string, number>
  confidence: number
}

export interface AnomalyEvaluationDetail {
  anomaly_score: number
  threshold: number
  threshold_tier: string
  is_anomaly: boolean
  calibrated_thresholds: Record<string, number>
}

export interface FlowEvaluationItem {
  flow_id: string
  destination_port?: number | null
  binary: BinaryEvaluationDetail
  multiclass: MulticlassEvaluationDetail
  anomaly: AnomalyEvaluationDetail
  consensus_status: string
  threat_status: string
  is_attack: boolean
  flow_summary: string
}

export interface AnalyzeFlowRequest {
  flows: Record<string, any>[]
  alert_on_threat_only?: boolean
  anomaly_threshold_tier?: string
}

export interface AnalyzeFlowResponse {
  alerts: AlertItem[]
  incidents: Incident[]
  evaluated_flows_count: number
  threats_count: number
  evaluations: FlowEvaluationItem[]
  models_used: string[]
  anomaly_threshold_tier: string
}

export interface FlowPresetItem {
  name: string
  category: string
  description: string
  flow: Record<string, any>
}

export interface FlowPresetsResponse {
  presets: Record<string, FlowPresetItem>
}
