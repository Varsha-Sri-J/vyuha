import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Activity,
  AlertCircle,
  ArrowUpRight,
  Bell,
  Bot,
  Check,
  ChevronDown,
  ChevronRight,
  Clock3,
  Eye,
  FileSearch,
  Filter,
  Info,
  LayoutDashboard,
  Lightbulb,
  Menu,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingUp,
  X,
  Zap,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type {
  AlertItem,
  AttentionItemData,
  Confidence,
  FlowTotals,
  Incident,
  IncidentEvidenceItem,
  IncidentStatus,
  InsightsData,
  NetworkActivityData,
  OverviewData,
  Page,
  SecurityAlertsData,
  Severity,
  TimelineEvent,
  Tone,
  TrafficWindow,
  WeekDayData,
} from './types/api'
import {
  checkApiHealth,
  getAlerts,
  getIncidentDetails,
  getIncidents,
  getInsights,
  getNetworkActivity,
  getOverview,
} from './services/api'

// ----------------------------------------------------------------------------
// Offline Fallback Baseline Constants (Shown only if backend is unreachable)
// ----------------------------------------------------------------------------

const initialTrafficData: TrafficWindow[] = [
  { name: '08:00', incoming: 28, outgoing: 20, unusual: 8 },
  { name: '10:00', incoming: 42, outgoing: 32, unusual: 12 },
  { name: '12:00', incoming: 37, outgoing: 26, unusual: 7 },
  { name: '14:00', incoming: 54, outgoing: 39, unusual: 16 },
  { name: '16:00', incoming: 47, outgoing: 44, unusual: 9 },
  { name: '18:00', incoming: 61, outgoing: 48, unusual: 22 },
  { name: '20:00', incoming: 51, outgoing: 35, unusual: 11 },
  { name: '22:00', incoming: 38, outgoing: 29, unusual: 6 },
]

const initialWeekData: WeekDayData[] = [
  { name: 'Mon', normal: 64, unusual: 8 },
  { name: 'Tue', normal: 72, unusual: 11 },
  { name: 'Wed', normal: 68, unusual: 7 },
  { name: 'Thu', normal: 81, unusual: 15 },
  { name: 'Fri', normal: 78, unusual: 12 },
  { name: 'Sat', normal: 44, unusual: 5 },
  { name: 'Sun', normal: 35, unusual: 3 },
]

const initialCategoryData = [
  { name: 'DoS_DDoS', value: 8, color: '#d47b62' },
  { name: 'PortScan', value: 6, color: '#89a596' },
  { name: 'BruteForce', value: 5, color: '#9b8ab4' },
  { name: 'WebAttack', value: 4, color: '#e2c866' },
]

const initialEngineData = [
  { name: 'Multiclass Classifier (RF)', matched: 90, reviewed: 100 },
  { name: 'Binary Intrusion Detector (RF)', matched: 99, reviewed: 100 },
  { name: 'Isolation Forest Anomaly Engine', matched: 80, reviewed: 100 },
  { name: 'Incident Grouping Engine', matched: 85, reviewed: 100 },
]

const initialAlerts: AlertItem[] = [
  {
    id: 'a-102',
    title: 'Outbound flow volume rose above baseline',
    category: 'DoS_DDoS',
    source: 'Binary intrusion detector',
    severity: 'Medium',
    summary: 'One sample window contains more outgoing flows than the surrounding trend.',
    time: 'Historical baseline',
    detail: 'Sample event: window contained higher outgoing flows compared with surrounding baseline.',
    evidence: 'The rise is visible against nearby windows in the offline evaluation partition.',
    whyFlagged: 'VYUHA marked the change for review because it differs from the baseline profile. It does not identify intent or confirm an attack.',
    technical: 'direction: outgoing · source: CICIDS2017-derived sample · IP/timestamp: unavailable',
    destination_port: 80,
  },
  {
    id: 'a-101',
    title: 'A short scan-like flow sequence stands out',
    category: 'PortScan',
    source: 'Multiclass classifier',
    severity: 'Low',
    summary: 'A short series of small inbound flows touched common service ports in the sample.',
    time: 'Historical baseline',
    detail: 'Sample event: inbound connections arrived on port 80 before stopping.',
    evidence: 'Supervised Random Forest classified flow as PortScan with high confidence.',
    whyFlagged: 'The rule engine found a pattern consistent with reconnaissance, but flow data alone cannot establish who initiated it or why.',
    technical: 'port: 80 · dataset: CICIDS2017 (offline) · IP context: not available',
    destination_port: 80,
  },
]

const initialIncidents: Incident[] = [
  {
    id: 'sample-inc-01',
    title: 'Candidate incident grouping flow signals on port 80',
    summary: 'VYUHA grouped nearby network-flow alerts sharing destination port 80 into a candidate incident for human review.',
    severity: 'High',
    alerts: 2,
    categories: ['PortScan', 'DoS_DDoS'],
    status: 'Review',
    updated: 'Updated recently',
    engines: ['Multiclass classifier', 'Binary intrusion detector'],
    relationship: 'The alerts share common destination port 80 and detection engine findings within the offline dataset record.',
    timeline: [
      { time: 'Phase 4', title: 'Candidate incident grouped', copy: 'Alerts grouped via shared destination port entity.', tone: 'sage' },
    ],
    nextSteps: ['Review service logs for port 80 to verify whether services accepted suspicious connections.', 'Close candidate if traffic was authorized administrative benchmarking.'],
    confidence: 'Moderate',
    evidence: [
      { icon: 'network', title: 'Target Port Focus (80)', copy: 'Observed alerts share destination port 80.', tone: 'terracotta' },
      { icon: 'activity', title: 'Cross-Engine Findings', copy: 'Supervised models flagged coinciding events on common services.', tone: 'sage' },
    ],
    missing_evidence: ['source_ip', 'destination_ip', 'packet_timestamps'],
  },
]

const navItems: { id: Page; label: string; icon: LucideIcon; short?: string }[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'activity', label: 'Network Activity', icon: Activity },
  { id: 'alerts', label: 'Security Alerts', icon: Bell, short: '4' },
  { id: 'incidents', label: 'Incidents', icon: Target, short: '2' },
  { id: 'details', label: 'Incident Details', icon: FileSearch },
  { id: 'insights', label: 'Insights', icon: Lightbulb },
]

const pathForPage: Record<Page, string> = {
  overview: '/',
  activity: '/network-activity',
  alerts: '/security-alerts',
  incidents: '/incidents',
  details: '/incident-details',
  insights: '/insights',
}

const iconMap: Record<string, LucideIcon> = {
  activity: Activity,
  network: Network,
  clock: Clock3,
  trending_up: TrendingUp,
  eye: Eye,
  zap: Zap,
  check: Check,
  info: Info,
}

function resolveIcon(icon: any): LucideIcon {
  if (typeof icon === 'string') {
    return iconMap[icon.toLowerCase()] || Info
  }
  return icon || Info
}

function resolveAttentionIcon(icon?: string): LucideIcon {
  const k = (icon || '').toLowerCase()
  if (k.includes('alert') || k.includes('terracotta')) return AlertCircle
  if (k.includes('zap') || k.includes('incident')) return Zap
  if (k.includes('eye') || k.includes('detail')) return Eye
  return Info
}

function pageForPath(pathname: string): Page {
  const match = (Object.entries(pathForPage) as [Page, string][]).find(([, path]) => path === pathname)
  return match?.[0] ?? 'overview'
}

const formatNumber = (value: number) => new Intl.NumberFormat('en-IN').format(value)

const severityStyles: Record<Severity, string> = {
  High: 'severity-high',
  Medium: 'severity-medium',
  Low: 'severity-low',
}

function statusStyles(status: IncidentStatus) {
  return `status-${status.toLowerCase().replaceAll(' ', '-')}`
}

// ----------------------------------------------------------------------------
// Root App Shell
// ----------------------------------------------------------------------------

function App() {
  const [activePage, setActivePage] = useState<Page>(() => pageForPath(window.location.pathname))
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [selectedIncident, setSelectedIncident] = useState<string | null>(null)

  // API State
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null)
  const [isSampleMode, setIsSampleMode] = useState(true)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // Page Data Stores
  const [overviewData, setOverviewData] = useState<OverviewData | null>(null)
  const [activityData, setActivityData] = useState<NetworkActivityData | null>(null)
  const [alertsData, setAlertsData] = useState<SecurityAlertsData | null>(null)
  const [incidentsData, setIncidentsData] = useState<IncidentsData | null>(null)
  const [insightsData, setInsightsData] = useState<InsightsData | null>(null)

  const navigate = (page: Page) => {
    setActivePage(page)
    window.history.pushState({}, '', pathForPage[page])
    setMobileNavOpen(false)
  }

  useEffect(() => {
    const handlePopState = () => setActivePage(pageForPath(window.location.pathname))
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  // Check health and load data for active page
  const loadPageData = useCallback(async (page: Page) => {
    setLoading(true)
    try {
      await checkApiHealth()
      setBackendConnected(true)
      setIsSampleMode(false)
      setConnectionError(null)

      if (page === 'overview') {
        const data = await getOverview()
        setOverviewData(data)
      } else if (page === 'activity') {
        const data = await getNetworkActivity()
        setActivityData(data)
      } else if (page === 'alerts') {
        const data = await getAlerts()
        setAlertsData(data)
      } else if (page === 'incidents') {
        const data = await getIncidents()
        setIncidentsData(data)
        if (data.incidents.length > 0) {
          // Dynamically ensure selectedIncident points to an existing incident
          setSelectedIncident((prev) => {
            if (prev && data.incidents.some((inc) => inc.id === prev)) {
              return prev
            }
            return data.incidents[0].id
          })
        }
      } else if (page === 'details') {
        // Direct navigation to /incident-details: fetch incidents list if not yet loaded
        if (!selectedIncident) {
          const data = await getIncidents()
          setIncidentsData(data)
          if (data.incidents.length > 0) {
            setSelectedIncident(data.incidents[0].id)
          }
        }
      } else if (page === 'insights') {
        const data = await getInsights()
        setInsightsData(data)
      }
    } catch (err: any) {
      setBackendConnected(false)
      setIsSampleMode(true)
      setConnectionError(err.message || 'Could not connect to FastAPI server at port 8000.')
    } finally {
      setLoading(false)
    }
  }, [selectedIncident])

  useEffect(() => {
    loadPageData(activePage)
  }, [activePage, loadPageData])

  return (
    <div className="app-shell">
      <Sidebar
        activePage={activePage}
        collapsed={!sidebarOpen}
        mobileOpen={mobileNavOpen}
        onNavigate={navigate}
        onToggle={() => setSidebarOpen((value) => !value)}
        onCloseMobile={() => setMobileNavOpen(false)}
        alertsCount={alertsData ? String(alertsData.total_count) : '4'}
        incidentsCount={incidentsData ? String(incidentsData.open_count) : '2'}
      />
      <main className={`main-content ${sidebarOpen ? '' : 'main-content-expanded'}`}>
        <Topbar
          onOpenMenu={() => setMobileNavOpen(true)}
          backendConnected={backendConnected}
          isSampleMode={isSampleMode}
        />
        <div className="page-wrap">
          <PrototypeBanner
            backendConnected={backendConnected}
            isSampleMode={isSampleMode}
            connectionError={connectionError}
            loading={loading}
            onRetry={() => loadPageData(activePage)}
          />

          {activePage === 'overview' && (
            <Overview
              onNavigate={navigate}
              data={overviewData}
              loading={loading}
            />
          )}

          {activePage === 'activity' && (
            <NetworkActivity
              data={activityData}
              loading={loading}
            />
          )}

          {activePage === 'alerts' && (
            <SecurityAlerts
              data={alertsData}
              loading={loading}
              onRefresh={(q, sev) => {
                getAlerts(q, sev)
                  .then(setAlertsData)
                  .catch(() => {})
              }}
            />
          )}

          {activePage === 'incidents' && (
            <IncidentsPage
              data={incidentsData}
              loading={loading}
              onOpenDetails={(id) => {
                setSelectedIncident(id)
                navigate('details')
              }}
              onFilterChange={(st) => {
                getIncidents(st)
                  .then(setIncidentsData)
                  .catch(() => {})
              }}
            />
          )}

          {activePage === 'details' && (
            <IncidentDetails
              incidentId={selectedIncident}
              fallbackIncidents={incidentsData?.incidents || initialIncidents}
              isSampleMode={isSampleMode}
              onBack={() => navigate('incidents')}
            />
          )}

          {activePage === 'insights' && (
            <Insights
              data={insightsData}
              loading={loading}
            />
          )}
        </div>
      </main>
    </div>
  )
}

// ----------------------------------------------------------------------------
// Navigation & Shell Components
// ----------------------------------------------------------------------------

function Sidebar({
  activePage,
  collapsed,
  mobileOpen,
  onNavigate,
  onToggle,
  onCloseMobile,
  alertsCount,
  incidentsCount,
}: {
  activePage: Page
  collapsed: boolean
  mobileOpen: boolean
  onNavigate: (page: Page) => void
  onToggle: () => void
  onCloseMobile: () => void
  alertsCount: string
  incidentsCount: string
}) {
  return (
    <>
      {mobileOpen && <button className="mobile-overlay" aria-label="Close navigation" onClick={onCloseMobile} />}
      <aside className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''} ${mobileOpen ? 'sidebar-mobile-open' : ''}`}>
        <div className="brand-row">
          <button className="brand-lockup" onClick={() => onNavigate('overview')} aria-label="Go to VYUHA overview">
            <span className="brand-mark"><span /></span>
            {!collapsed && <span className="brand-name">VYUHA</span>}
          </button>
          <button className="mobile-close" onClick={onCloseMobile} aria-label="Close navigation"><X size={18} /></button>
        </div>

        {!collapsed && (
          <div className="workspace-switcher">
            <div className="workspace-icon">A</div>
            <div className="workspace-copy"><strong>Aranya Foods</strong><span>Security Operations</span></div>
            <span className="sample-workspace-label">FLOW INTELLIGENCE</span>
          </div>
        )}

        <div className="nav-section-label">Workspace</div>
        <nav className="primary-nav" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = activePage === item.id
            const countLabel = item.id === 'alerts' ? alertsCount : item.id === 'incidents' ? incidentsCount : item.short
            return (
              <button
                key={item.id}
                className={`nav-item ${isActive ? 'nav-item-active' : ''}`}
                onClick={() => onNavigate(item.id)}
                aria-current={isActive ? 'page' : undefined}
                title={collapsed ? item.label : undefined}
              >
                <Icon size={18} strokeWidth={isActive ? 2.2 : 1.8} />
                {!collapsed && <span>{item.label}</span>}
                {!collapsed && countLabel && (
                  <span className={`nav-count ${item.id === 'alerts' ? 'nav-count-alert' : ''}`}>
                    {countLabel}
                  </span>
                )}
              </button>
            )
          })}
        </nav>

        <div className="sidebar-bottom">
          <div className="profile-row" aria-label="Current workspace owner">
            <span className="avatar avatar-small">VS</span>
            {!collapsed && <span className="profile-copy"><strong>Varsha</strong><span>Security Lead</span></span>}
          </div>
          <button className="collapse-button" onClick={onToggle} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>
        </div>
      </aside>
    </>
  )
}

function Topbar({
  onOpenMenu,
  backendConnected,
  isSampleMode,
}: {
  onOpenMenu: () => void
  backendConnected: boolean | null
  isSampleMode: boolean
}) {
  return (
    <header className="topbar">
      <button className="mobile-menu" onClick={onOpenMenu} aria-label="Open navigation"><Menu size={20} /></button>
      <div className="breadcrumb"><span>Workspace</span><ChevronRight size={14} /><strong>Security overview</strong></div>
      <div className="topbar-actions">
        {backendConnected ? (
          <div className="sample-pill sample-pill-connected" title="Connected to local VYUHA backend API (offline benchmark telemetry)">
            <span className="pulse-dot" /> Backend API connected
          </div>
        ) : (
          <div className="sample-pill sample-pill-offline" title={isSampleMode ? 'Backend unreachable: displaying fallback sample data' : 'Checking backend...'}>
            <span className="pulse-dot pulse-dot-amber" /> Offline sample mode
          </div>
        )}
        <span className="avatar">VS</span>
      </div>
    </header>
  )
}

function PrototypeBanner({
  backendConnected,
  isSampleMode,
  connectionError,
  loading,
  onRetry,
}: {
  backendConnected: boolean | null
  isSampleMode: boolean
  connectionError: string | null
  loading: boolean
  onRetry: () => void
}) {
  if (backendConnected) {
    return (
      <div className="prototype-banner prototype-banner-connected">
        <div className="prototype-icon" style={{ background: '#dbe8de', color: '#527560' }}><ShieldCheck size={15} /></div>
        <span><strong>Backend API Connected</strong> · Telemetry served from local VYUHA Incident Intelligence API. Data source: Offline CICIDS2017 benchmark evaluations and demonstration flow models (not live packet capture).</span>
        {loading && <span className="loading-badge" style={{ marginLeft: 'auto' }}><RefreshCw size={12} className="animate-spin" /> Fetching...</span>}
      </div>
    )
  }

  return (
    <div className="prototype-banner prototype-banner-error">
      <div className="prototype-icon" style={{ background: '#f5dfd7', color: '#b25e48' }}><AlertCircle size={15} /></div>
      <span>
        <strong>Offline Fallback Mode</strong> · {connectionError ? `API unreachable (${connectionError}).` : 'Backend API offline.'} Showing illustrative sample flow patterns. Run `uvicorn src.api.app:app` to connect the backend API.
      </span>
      <button className="retry-button" onClick={onRetry} disabled={loading}>
        <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> {loading ? 'Checking...' : 'Retry connection'}
      </button>
    </div>
  )
}

function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string
  title: string
  description: string
  actions?: ReactNode
}) {
  return (
    <div className="page-header">
      <div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  )
}

// ----------------------------------------------------------------------------
// 1. Overview Page
// ----------------------------------------------------------------------------

function Overview({
  onNavigate,
  data,
  loading,
}: {
  onNavigate: (page: Page) => void
  data: OverviewData | null
  loading: boolean
}) {
  const traffic = data?.traffic_data || initialTrafficData
  const alertsList = data?.recent_alerts || initialAlerts.slice(0, 3)

  const totalObserved = data?.flow_totals?.total_observed ?? 199990
  const suspiciousCount = data?.flow_totals?.suspicious ?? 34067
  const categoriesCount = data?.categories_count ?? initialCategoryData.length
  const openIncidents = data?.open_incidents_count ?? 2

  const attentionItems: AttentionItemData[] = (data?.attention_items && data.attention_items.length > 0)
    ? data.attention_items
    : [
        {
          title: 'Candidate incidents active for review',
          copy: 'Correlated across destination ports and detection engines for review.',
          tone: 'yellow' as Tone,
          target: 'incidents' as Page,
          icon: 'zap',
        },
        {
          title: 'Coinciding flow deviations flagged on port 80',
          copy: 'Supervised and anomaly models flagged coinciding events on common services.',
          tone: 'terracotta' as Tone,
          target: 'alerts' as Page,
          icon: 'alert',
        },
        {
          title: 'Contextual review required',
          copy: 'Offline benchmark dataset lacks host identities; review context before drawing conclusions.',
          tone: 'lavender' as Tone,
          target: 'details' as Page,
          icon: 'eye',
        },
      ]

  return (
    <>
      <PageHeader
        eyebrow="Thursday, 19 November 2026"
        title="A calm view of your network flows"
        description="Review traffic baseline, suspicious flow patterns, and candidate incidents without treating any signal as an unverified verdict."
      />
      <div className="overview-grid">
        <section className="hero-card card-lift">
          <div className="hero-content">
            <div className="section-kicker"><span className="status-dot status-dot-green" /> Network posture</div>
            <div className="hero-number">{data?.posture || 'Steady'} <span>·</span> <em>{data?.posture_detail || 'baseline view'}</em></div>
            <p>{data?.posture_summary || 'The offline flow distribution is consistent with validated CICIDS2017 benchmark evaluations.'}</p>
            <button className="text-button" onClick={() => onNavigate('activity')}>View network activity <ArrowUpRight size={15} /></button>
          </div>
          <NetworkPulse />
        </section>

        <div className="stat-card-grid">
          <StatCard
            icon={Network}
            label="Total flows observed"
            value={formatNumber(totalObserved)}
            helper="Validated dataset partition"
            iconClass="icon-sage"
            trend="up"
          />
          <StatCard
            icon={Bell}
            label="Flagged flow patterns"
            value={formatNumber(suspiciousCount)}
            helper="Review signal, not a verdict"
            iconClass="icon-terracotta"
            trend="attention"
          />
          <StatCard
            icon={Target}
            label="Attack / anomaly categories"
            value={String(categoriesCount)}
            helper="Phase 3C multiclass families"
            iconClass="icon-lavender"
            trend="neutral"
          />
          <StatCard
            icon={TrendingUp}
            label="Open candidate incidents"
            value={String(openIncidents)}
            helper="Phase 4 correlated groups"
            iconClass="icon-yellow"
            trend="neutral"
          />
        </div>
      </div>

      <div className="content-grid overview-content-grid">
        <section className="card activity-card">
          <CardHeading
            title="Recent network activity"
            subtitle="Flow volume aggregated across observation windows"
            action={<button className="quiet-button" onClick={() => onNavigate('activity')}>Explore <ChevronRight size={15} /></button>}
          />
          <div className="chart-legend">
            <span><i className="legend-dot legend-sage" /> Observed flows</span>
            <span><i className="legend-dot legend-terracotta" /> Suspicious pattern</span>
            <span className="legend-caption">Diurnal window · 24h</span>
          </div>
          <div className="chart-wrap chart-wrap-large">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={traffic} margin={{ top: 10, right: 8, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="normalFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#8eaa99" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#8eaa99" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="unusualFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#d47b62" stopOpacity={0.24} />
                    <stop offset="100%" stopColor="#d47b62" stopOpacity={0.01} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="#e7e1d7" strokeDasharray="4 5" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9b958c', fontSize: 11 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#9b958c', fontSize: 11 }} />
                <Tooltip content={<ChartTooltip />} />
                <Area type="monotone" dataKey="incoming" name="Observed flows" stroke="#89a596" strokeWidth={2.5} fill="url(#normalFill)" />
                <Area type="monotone" dataKey="unusual" name="Suspicious pattern" stroke="#d47b62" strokeWidth={2.5} fill="url(#unusualFill)" />
                <Line type="monotone" dataKey="incoming" name="Observed flows" stroke="#89a596" strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="unusual" name="Suspicious pattern" stroke="#d47b62" strokeWidth={2.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card attention-card">
          <CardHeading
            title="What needs attention"
            subtitle="Prioritized starting points based on model evaluation"
            action={<span className="attention-count">{attentionItems.length} prioritized</span>}
          />
          <div className="attention-list">
            {attentionItems.map((item, idx) => {
              const IconComp = resolveAttentionIcon(item.icon || item.type)
              const tone = item.tone || (idx === 0 ? 'yellow' : idx === 1 ? 'terracotta' : 'lavender')
              const target = (item.target || (idx === 0 ? 'incidents' : idx === 1 ? 'alerts' : 'details')) as Page
              return (
                <AttentionItem
                  key={idx}
                  icon={IconComp}
                  tone={tone}
                  title={item.title}
                  copy={item.copy}
                  onClick={() => onNavigate(target)}
                />
              )
            })}
          </div>
          <div className="attention-footer"><Check size={15} /> Evaluated from offline benchmark distribution; human context required.</div>
        </section>
      </div>

      <div className="content-grid lower-grid">
        <section className="card alert-preview-card">
          <CardHeading title="Latest security alerts" subtitle="Normalized flow signals from all detection engines" action={<button className="quiet-button" onClick={() => onNavigate('alerts')}>See all <ChevronRight size={15} /></button>} />
          <div className="table-list">
            {alertsList.slice(0, 3).map((alert) => (
              <AlertRow key={alert.id} alert={alert} mode="navigate" onToggle={() => onNavigate('alerts')} />
            ))}
          </div>
        </section>
        <section className="card quick-card">
          <div className="quick-card-shape shape-one" /><div className="quick-card-shape shape-two" />
          <div className="section-kicker"><Sparkles size={15} /> Incident intelligence</div>
          <h3>Security gets easier when the flow story is clear.</h3>
          <p>VYUHA brings related network signals together into explainable candidate incidents, without pretending offline data is live telemetry.</p>
          <button className="button button-dark" onClick={() => onNavigate('insights')}>Explore your insights <ArrowUpRight size={16} /></button>
        </section>
      </div>
    </>
  )
}

function NetworkPulse() {
  return (
    <div className="network-visual" aria-label="Illustrative flow pattern">
      <div className="network-glow" />
      <svg viewBox="0 0 360 220" role="img" aria-hidden="true">
        <path className="network-line line-one" d="M55 118 C104 44 137 93 180 65 S252 54 309 100" />
        <path className="network-line line-two" d="M48 154 C106 181 130 112 180 128 S247 176 311 137" />
        <path className="network-line line-three" d="M87 42 C119 95 157 99 180 65 S210 31 254 74" />
        <circle className="network-node node-main" cx="180" cy="101" r="27" /><circle className="network-node node-small" cx="55" cy="118" r="8" /><circle className="network-node node-small" cx="309" cy="100" r="8" /><circle className="network-node node-small" cx="87" cy="42" r="7" /><circle className="network-node node-small" cx="311" cy="137" r="7" />
        <circle className="network-node node-pulse" cx="180" cy="101" r="37" />
      </svg>
      <div className="network-label label-main"><ShieldCheck size={14} /> baseline view</div>
      <div className="network-label label-device">flow distribution</div>
    </div>
  )
}

function StatCard({ icon: Icon, label, value, helper, iconClass, trend }: { icon: LucideIcon; label: string; value: string; helper: string; iconClass: string; trend: 'up' | 'attention' | 'neutral' }) {
  return (
    <div className="stat-card card-lift">
      <div className={`stat-icon ${iconClass}`}><Icon size={17} /></div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className={`stat-helper stat-helper-${trend}`}>
        {trend === 'up' && <ArrowUpRight size={13} />}
        {trend === 'attention' && <span className="mini-dot" />}
        {helper}
      </div>
    </div>
  )
}

function AttentionItem({ icon: Icon, tone, title, copy, onClick }: { icon: LucideIcon; tone: string; title: string; copy: string; onClick: () => void }) {
  return (
    <button className="attention-item" onClick={onClick}>
      <span className={`attention-icon attention-${tone}`}><Icon size={16} /></span>
      <span className="attention-copy"><strong>{title}</strong><span>{copy}</span></span>
      <ChevronRight size={16} className="muted-icon" />
    </button>
  )
}

function CardHeading({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return <div className="card-heading"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <span>{label}</span>
      {payload.map((item) => (
        <div key={item.name}>
          <i style={{ backgroundColor: item.color }} />
          {item.name}
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  )
}

// ----------------------------------------------------------------------------
// 2. Network Activity Page
// ----------------------------------------------------------------------------

function NetworkActivity({
  data,
  loading,
}: {
  data: NetworkActivityData | null
  loading: boolean
}) {
  const traffic = data?.traffic_data || initialTrafficData
  const week = data?.week_data || initialWeekData
  const moments = data?.moments || [
    { time: '18:00', title: 'Outgoing volume rose above baseline', copy: 'Flow rate and packet size distributions deviated from the median baseline.' },
    { time: '14:00', title: 'Destination port scan sequence flagged', copy: 'Supervised Random Forest classified sequence as PortScan with 98.5% confidence.' },
    { time: '10:00', title: 'Isolation Forest out-of-distribution detection', copy: 'Unsupervised anomaly model scored flow at 0.742 against calibrated threshold 0.584.' },
  ]

  const total = data?.flow_totals?.total_observed ?? 199990
  const incoming = data?.flow_totals?.incoming ?? 115994
  const outgoing = data?.flow_totals?.outgoing ?? 83996
  const suspicious = data?.flow_totals?.suspicious ?? 34067

  return (
    <>
      <PageHeader
        eyebrow="Network activity"
        title="A clearer view of your network flows"
        description="See traffic volume, direction, and unusual windows without adding measurements the current dataset cannot support."
      />
      <div className="activity-stat-row">
        <MiniMetric label="Total flows observed" value={formatNumber(total)} note="Dataset evaluation volume" tone="sage" />
        <MiniMetric label="Incoming flow estimate" value={formatNumber(incoming)} note="Forward packet distribution" tone="lavender" />
        <MiniMetric label="Outgoing flow estimate" value={formatNumber(outgoing)} note="Backward packet distribution" tone="yellow" />
        <MiniMetric label="Flagged flow patterns" value={formatNumber(suspicious)} note="Review signal, not a verdict" tone="terracotta" />
      </div>

      <section className="card big-chart-card">
        <CardHeading
          title="Traffic through observation windows"
          subtitle="Flow volume aggregated by window"
          action={
            <div className="chart-legend compact">
              <span><i className="legend-dot legend-sage" /> Incoming</span>
              <span><i className="legend-dot legend-lavender" /> Outgoing</span>
              <span><i className="legend-dot legend-terracotta" /> Suspicious</span>
            </div>
          }
        />
        <div className="chart-wrap chart-wrap-xl">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={traffic} margin={{ top: 14, right: 16, left: -12, bottom: 0 }}>
              <defs>
                <linearGradient id="incomingFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#89a596" stopOpacity={0.28} />
                  <stop offset="100%" stopColor="#89a596" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="outgoingFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#9b8ab4" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#9b8ab4" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} stroke="#e7e1d7" strokeDasharray="4 5" />
              <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9b958c', fontSize: 11 }} />
              <YAxis axisLine={false} tickLine={false} tick={{ fill: '#9b958c', fontSize: 11 }} />
              <Tooltip content={<ChartTooltip />} />
              <Area type="monotone" dataKey="incoming" name="Incoming" stroke="#89a596" fill="url(#incomingFill)" strokeWidth={2.5} />
              <Area type="monotone" dataKey="outgoing" name="Outgoing" stroke="#9b8ab4" fill="url(#outgoingFill)" strokeWidth={2.5} />
              <Line type="monotone" dataKey="unusual" name="Suspicious" stroke="#d47b62" strokeWidth={2.5} dot={{ r: 3, fill: '#d47b62', strokeWidth: 0 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-note">
          <Info size={14} /> {data?.disclaimer || 'Capacity, latency, packet loss, and congestion are not available in this dataset.'}
        </div>
      </section>

      <div className="content-grid activity-lower-grid">
        <section className="card">
          <CardHeading title="Activity partition view" subtitle="Observed normal vs flagged counts" />
          <div className="chart-wrap chart-wrap-medium">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={week} barGap={7} margin={{ top: 12, right: 4, left: -22, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="#e7e1d7" strokeDasharray="4 5" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9b958c', fontSize: 11 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#9b958c', fontSize: 11 }} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="normal" name="Observed" fill="#89a596" radius={[5, 5, 2, 2]} />
                <Bar dataKey="unusual" name="Suspicious" fill="#d47b62" radius={[5, 5, 2, 2]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card">
          <CardHeading title="Moments worth a look" subtitle="Flow context is more useful than alarm" />
          <div className="moment-list">
            {moments.map((m, idx) => (
              <MomentItem key={idx} time={m.time} title={m.title} copy={m.copy} />
            ))}
          </div>
        </section>
      </div>
    </>
  )
}

function MiniMetric({ label, value, note, tone }: { label: string; value: string; note: string; tone: string }) {
  return <div className={`mini-metric mini-metric-${tone}`}><span>{label}</span><strong>{value}</strong><small>{note}</small></div>
}

function MomentItem({ time, title, copy }: { time: string; title: string; copy: string }) {
  return <div className="moment-item"><span className="moment-time">{time}</span><div><strong>{title}</strong><p>{copy}</p></div></div>
}

// ----------------------------------------------------------------------------
// 3. Security Alerts Page
// ----------------------------------------------------------------------------

function SecurityAlerts({
  data,
  loading,
  onRefresh,
}: {
  data: SecurityAlertsData | null
  loading: boolean
  onRefresh: (query?: string, severity?: string) => void
}) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<'All' | Severity>('All')
  const [expanded, setExpanded] = useState<string | null>(null)
  const [reviewed, setReviewed] = useState<string[]>([])
  const [saved, setSaved] = useState(false)
  const [showApproach, setShowApproach] = useState(false)

  const alertsList = data?.alerts || initialAlerts

  const filtered = useMemo(() => {
    return alertsList.filter((alert) => {
      const matchesFilter = filter === 'All' || alert.severity === filter
      const text = `${alert.title} ${alert.category} ${alert.summary}`.toLowerCase()
      const matchesQuery = text.includes(query.toLowerCase())
      return matchesFilter && matchesQuery
    })
  }, [alertsList, filter, query])

  const totalCount = data?.total_count ?? alertsList.length
  const suspiciousCount = data?.suspicious_flow_count ?? 34067

  return (
    <>
      <PageHeader
        eyebrow="Security alerts"
        title="Signals, explained simply"
        description="These are network-flow moments noticed as deviations from the baseline. They are prompts for review, not confirmed attacks."
        actions={
          <button className={`button button-secondary ${saved ? 'button-saved' : ''}`} onClick={() => setSaved((v) => !v)}>
            <Filter size={16} /> {saved ? 'View saved' : 'Save view'}
          </button>
        }
      />
      <div className="alert-overview-strip">
        <div><span className="strip-label">Signals available</span><strong>{totalCount}</strong><span>evaluated</span></div>
        <div className="strip-divider" />
        <div><span className="strip-label">Flagged flow patterns</span><strong>{formatNumber(suspiciousCount)}</strong><span>in benchmark partition</span></div>
        <div className="strip-action"><ShieldCheck size={18} /> No alert is a verdict without context</div>
      </div>

      <section className="card alerts-card">
        <div className="alerts-toolbar">
          <div className="search-input">
            <Search size={17} />
            <input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
              }}
              placeholder="Search alerts"
              aria-label="Search alerts"
            />
            {query && <button onClick={() => setQuery('')} aria-label="Clear search"><X size={15} /></button>}
          </div>

          <div className="filter-pills">
            {(['All', 'High', 'Medium', 'Low'] as const).map((item) => (
              <button
                key={item}
                className={`filter-pill ${filter === item ? 'filter-pill-active' : ''}`}
                onClick={() => setFilter(item)}
              >
                {item}
                {item !== 'All' && <span className={`filter-mini-dot ${severityStyles[item]}`} />}
              </button>
            ))}
          </div>

          <span className="result-count">{filtered.length} signals</span>
        </div>

        <div className="alert-list">
          {filtered.map((alert) => (
            <AlertRow
              key={alert.id}
              alert={alert}
              mode="expand"
              expanded={expanded === alert.id}
              reviewed={reviewed.includes(alert.id)}
              onToggle={() => setExpanded(expanded === alert.id ? null : alert.id)}
              onReview={() => setReviewed((items) => items.includes(alert.id) ? items.filter((id) => id !== alert.id) : [...items, alert.id])}
            />
          ))}
          {filtered.length === 0 && (
            <div className="empty-state">
              <Search size={26} />
              <strong>No signals match that search</strong>
              <span>Try a different word or clear the filter.</span>
            </div>
          )}
        </div>
      </section>

      <div className="technical-note">
        <Bot size={17} />
        <div>
          <strong>How VYUHA normalizes flow signals</strong>
          <span>Combines binary, multiclass, and unsupervised anomaly engines into canonical Alert representations. An alert is an invitation to review, not a verdict.</span>
          {showApproach && (
            <span className="approach-detail">
              The CICIDS2017 benchmark dataset contains flow volume, duration, and destination port. IP addresses, protocols, and host identities are not present in the offline feature set.
            </span>
          )}
        </div>
        <button className="quiet-button" onClick={() => setShowApproach((v) => !v)}>
          {showApproach ? 'Hide approach' : 'Read the approach'} <ChevronRight size={15} />
        </button>
      </div>
    </>
  )
}

function AlertRow({
  alert,
  mode,
  expanded = false,
  reviewed = false,
  onToggle,
  onReview,
}: {
  alert: AlertItem
  mode: 'expand' | 'navigate'
  expanded?: boolean
  reviewed?: boolean
  onToggle?: () => void
  onReview?: () => void
}) {
  const rowContent = (
    <>
      <span className={`severity-badge ${severityStyles[alert.severity]}`}><span />{alert.severity}</span>
      <span className="alert-row-copy"><strong>{alert.title}</strong><span>{alert.summary}</span></span>
      <span className="alert-row-meta"><span>{alert.category}</span><time>{alert.time}</time></span>
      {reviewed && <span className="reviewed-label"><Check size={12} /> reviewed</span>}
      {mode === 'navigate' ? <ArrowUpRight size={17} className="row-chevron" /> : <ChevronDown size={17} className={`row-chevron ${expanded ? 'row-chevron-open' : ''}`} />}
    </>
  )

  return (
    <div className={`alert-row ${expanded ? 'alert-row-expanded' : ''} ${reviewed ? 'alert-row-reviewed' : ''}`}>
      <button className={`alert-row-main ${mode === 'navigate' ? 'alert-row-main-navigate' : ''}`} onClick={onToggle} aria-expanded={mode === 'expand' ? expanded : undefined}>
        {rowContent}
      </button>
      {expanded && (
        <div className="alert-expanded">
          <div><span className="detail-label">What the flow showed</span><p>{alert.detail}</p></div>
          <div><span className="detail-label">Relevant evidence</span><p>{alert.evidence}</p></div>
          <div><span className="detail-label">Detection engine</span><strong>{alert.source}</strong></div>
          <div><span className="detail-label">Why it was flagged</span><p>{alert.whyFlagged}</p></div>
          <div><span className="detail-label">Technical detail</span><p>{alert.technical}</p></div>
          <div className="expanded-actions">
            <button className="button button-small button-secondary" onClick={(e) => { e.stopPropagation(); onReview?.() }}>
              {reviewed ? 'Mark as unreviewed' : 'Mark as reviewed'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ----------------------------------------------------------------------------
// 4. Incidents Page
// ----------------------------------------------------------------------------

function IncidentsPage({
  data,
  loading,
  onOpenDetails,
  onFilterChange,
}: {
  data: IncidentsData | null
  loading: boolean
  onOpenDetails: (id: string) => void
  onFilterChange: (status: string) => void
}) {
  const incidentsList = data?.incidents || initialIncidents
  const [expanded, setExpanded] = useState<string | null>(() => incidentsList[0]?.id || null)
  const [filter, setFilter] = useState<'All' | IncidentStatus>('All')
  const [filterOpen, setFilterOpen] = useState(false)
  const [showGroupingHelp, setShowGroupingHelp] = useState(false)

  const filteredIncidents = useMemo(() => {
    return incidentsList.filter((incident) => filter === 'All' || incident.status === filter)
  }, [incidentsList, filter])

  return (
    <>
      <PageHeader
        eyebrow="Candidate incidents"
        title="The bigger picture"
        description="VYUHA groups related network-flow alerts into candidate incidents, so you can review a coherent story instead of a pile of isolated signals."
        actions={
          <button className={`button button-secondary ${filter !== 'All' ? 'button-saved' : ''}`} onClick={() => setFilterOpen((v) => !v)}>
            <Filter size={16} /> {filter === 'All' ? 'Filter incidents' : filter}
          </button>
        }
      />

      {filterOpen && (
        <div className="incident-filter-menu" role="group" aria-label="Filter incidents">
          {(['All', 'Review', 'Monitoring', 'Closed candidate'] as const).map((item) => (
            <button
              key={item}
              className={`filter-pill ${filter === item ? 'filter-pill-active' : ''}`}
              onClick={() => {
                setFilter(item)
                setFilterOpen(false)
                onFilterChange(item)
              }}
            >
              {item}
            </button>
          ))}
        </div>
      )}

      <div className="incident-intro">
        <div className="incident-intro-icon"><Sparkles size={22} /></div>
        <div>
          <strong>What is a candidate incident?</strong>
          <span>A possible story constructed from correlated network-flow events sharing target endpoints or detection engines. It is not an automated verdict. You decide what to do next.</span>
        </div>
        <button className="quiet-button" onClick={() => setShowGroupingHelp((v) => !v)}>
          {showGroupingHelp ? 'Hide grouping guide' : 'How grouping works'} <ChevronRight size={15} />
        </button>
      </div>

      {showGroupingHelp && (
        <div className="grouping-help">
          <strong>Phase 4 Grouping Architecture</strong>
          <span>VYUHA constructs a lightweight RelationshipGraph connecting alerts via shared destination ports, same flow IDs, attack categories, and detection-engine agreement. Temporal correlation links alerts within configurable time windows. Unrelated alerts remain separate.</span>
        </div>
      )}

      <div className="incident-list">
        {filteredIncidents.map((incident) => (
          <IncidentCard
            key={incident.id}
            incident={incident}
            expanded={expanded === incident.id}
            onToggle={() => setExpanded(expanded === incident.id ? null : incident.id)}
            onOpenDetails={() => onOpenDetails(incident.id)}
          />
        ))}
        {filteredIncidents.length === 0 && (
          <div className="empty-state">
            <Target size={26} />
            <strong>No candidate incidents in this view</strong>
            <span>Choose another status filter.</span>
          </div>
        )}
      </div>
    </>
  )
}

function IncidentCard({
  incident,
  expanded,
  onToggle,
  onOpenDetails,
}: {
  incident: Incident
  expanded: boolean
  onToggle: () => void
  onOpenDetails: () => void
}) {
  return (
    <section className={`card incident-card ${expanded ? 'incident-card-open' : ''}`}>
      <button className="incident-card-head" onClick={onToggle} aria-expanded={expanded}>
        <div className={`incident-symbol ${severityStyles[incident.severity]}`}>
          <Target size={19} />
        </div>
        <div className="incident-title">
          <div>
            <span className="incident-id">{incident.id}</span>
            <span className="candidate-chip">Candidate incident</span>
            <span className={`status-chip ${statusStyles(incident.status)}`}>{incident.status}</span>
          </div>
          <h2>{incident.title}</h2>
          <p>{incident.summary}</p>
        </div>
        <div className="incident-meta">
          <span className={`severity-badge ${severityStyles[incident.severity]}`}><span />{incident.severity}</span>
          <span className="related-count"><strong>{incident.alerts}</strong> related alerts</span>
          <ChevronDown size={18} className={`row-chevron ${expanded ? 'row-chevron-open' : ''}`} />
        </div>
      </button>

      {expanded && (
        <div className="incident-card-body">
          <div className="grouping-story">
            <span className="detail-label">Why VYUHA grouped these events</span>
            <p>{incident.relationship || incident.summary}</p>
            <div className="category-row">
              {incident.categories.map((cat) => (
                <span key={cat} className="category-tag">{cat}</span>
              ))}
            </div>
            <div className="engine-tags">
              <span className="detail-label">Detection engines involved</span>
              <div>
                {incident.engines.map((eng) => (
                  <span key={eng} className="engine-tag">{eng}</span>
                ))}
              </div>
            </div>
          </div>

          <div className="incident-next">
            <span className="detail-label">Recommended next step</span>
            <strong>{incident.nextSteps[0] || 'Review destination ports and verify expected traffic.'}</strong>
            <span className="incident-timeline-note">
              {incident.timeline.length} corroborating signals · {incident.confidence} connection
            </span>
            <button className="button button-small button-dark" onClick={(e) => { e.stopPropagation(); onOpenDetails() }}>
              Open incident details <ArrowUpRight size={14} />
            </button>
          </div>
        </div>
      )}
    </section>
  )
}

// ----------------------------------------------------------------------------
// 5. Incident Details Page
// ----------------------------------------------------------------------------

function IncidentDetails({
  incidentId,
  fallbackIncidents,
  isSampleMode,
  onBack,
}: {
  incidentId: string | null
  fallbackIncidents: Incident[]
  isSampleMode: boolean
  onBack: () => void
}) {
  const [incident, setIncident] = useState<Incident | null>(() => {
    if (incidentId) {
      return fallbackIncidents.find((i) => i.id === incidentId) || null
    }
    return fallbackIncidents[0] || null
  })
  const [technicalOpen, setTechnicalOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [errorNotFound, setErrorNotFound] = useState(false)

  useEffect(() => {
    let isCurrent = true
    const targetId = incidentId || fallbackIncidents[0]?.id
    if (!targetId) {
      setErrorNotFound(true)
      return
    }

    setLoading(true)
    setErrorNotFound(false)
    getIncidentDetails(targetId)
      .then((data) => {
        if (isCurrent && data) {
          setIncident(data)
          setErrorNotFound(false)
        }
      })
      .catch((err) => {
        if (!isCurrent) return
        // If in sample mode (backend offline), check fallback incidents
        if (isSampleMode) {
          const fb = fallbackIncidents.find((i) => i.id === targetId)
          if (fb) {
            setIncident(fb)
            return
          }
        }
        // When connected to API and incident not found, NEVER silently substitute!
        setIncident(null)
        setErrorNotFound(true)
      })
      .finally(() => {
        if (isCurrent) setLoading(false)
      })

    return () => {
      isCurrent = false
    }
  }, [incidentId, fallbackIncidents, isSampleMode])

  if (loading) {
    return (
      <div className="empty-state" style={{ padding: '60px 0' }}>
        <RefreshCw size={26} className="animate-spin" />
        <strong>Loading candidate incident...</strong>
        <span>Querying VYUHA Incident Intelligence backend.</span>
      </div>
    )
  }

  if (errorNotFound || !incident) {
    return (
      <div className="empty-state" style={{ padding: '60px 0' }}>
        <Target size={30} />
        <strong>Candidate incident not found</strong>
        <span>
          {incidentId
            ? `Incident "${incidentId}" does not exist in the current evaluation partition.`
            : 'No candidate incident is currently selected.'}
        </span>
        <button className="button button-small button-secondary" style={{ marginTop: 12 }} onClick={onBack}>
          <ChevronRight size={15} className="back-icon" /> Return to incidents list
        </button>
      </div>
    )
  }

  return (
    <>
      <button className="back-link" onClick={onBack}>
        <ChevronRight size={15} className="back-icon" /> Back to incidents
      </button>

      <PageHeader
        eyebrow={`Investigation · ${incident.id}`}
        title={incident.title}
        description="A readable investigation dossier for a candidate incident. No automated conclusion has been made."
        actions={
          <div className="detail-status-actions">
            <span className="candidate-chip">Candidate incident</span>
            <span className={`status-chip ${statusStyles(incident.status)} status-chip-large`}>{incident.status}</span>
          </div>
        }
      />

      <div className="detail-layout">
        <div className="detail-main">
          <section className="card story-card">
            <div className="story-card-header">
              <div className={`incident-symbol ${severityStyles[incident.severity]}`}>
                <Target size={20} />
              </div>
              <div>
                <span className="detail-label">What happened?</span>
                <h2>{incident.title}</h2>
              </div>
            </div>
            <p>{incident.relationship?.trim() || incident.summary}</p>
            <div className="confidence-row">
              <div>
                <span>Connection strength</span>
                <strong>{incident.confidence}</strong>
              </div>
              <div className="confidence-bar">
                <span style={{ width: incident.confidence === 'Strong' ? '82%' : incident.confidence === 'Moderate' ? '56%' : '32%' }} />
              </div>
              <span className="muted-copy">Requires analyst context</span>
            </div>
          </section>

          <section className="card">
            <CardHeading title="Why was it flagged?" subtitle="The flow signals and corroborating evidence" />
            {incident.evidence && incident.evidence.length > 0 ? (
              <div className="evidence-grid">
                {incident.evidence.map((item, idx) => (
                  <EvidenceItem key={idx} {...item} />
                ))}
              </div>
            ) : (
              <div className="empty-state" style={{ padding: '24px 0' }}>
                <Info size={20} />
                <span>No corroborating multi-endpoint evidence available in offline flow records.</span>
              </div>
            )}
          </section>

          <section className="card">
            <CardHeading title="Corroboration timeline" subtitle="Signals and observations within the analysis window" />
            {incident.timeline && incident.timeline.length > 0 ? (
              <div className="timeline">
                {incident.timeline.map((item, idx) => (
                  <TimelineItem key={idx} {...item} />
                ))}
              </div>
            ) : (
              <div className="empty-state" style={{ padding: '24px 0' }}>
                <Clock3 size={20} />
                <span>No chronological sequence available in offline flow records.</span>
              </div>
            )}
          </section>
        </div>

        <aside className="detail-aside">
          <section className="card aside-card">
            <div className="aside-eyebrow"><Bot size={15} /> Detection engines</div>
            <h3>{incident.engines.length} engine view{incident.engines.length > 1 ? 's' : ''}</h3>
            <p>Different flow analysis views observed different aspects of the flow event.</p>
            <div className="engine-list">
              {incident.engines.map((engine, idx) => (
                <EngineLine
                  key={engine}
                  name={engine}
                  value={idx === 0 ? 'Primary' : idx === 1 ? 'Corroborating' : 'Context'}
                  color={idx === 0 ? 'sage' : idx === 1 ? 'lavender' : 'yellow'}
                />
              ))}
            </div>
            <button className="quiet-button quiet-button-full" onClick={() => setTechnicalOpen((v) => !v)}>
              {technicalOpen ? 'Hide technical detail' : 'See technical detail'} <ChevronRight size={15} />
            </button>
            {technicalOpen && (
              <div className="technical-detail">
                Evaluated against offline CICIDS2017 flow distributions. Features include packet asymmetries, destination port, and duration. Source IP, destination IP, and real-time socket timestamps are not available in this dataset.
              </div>
            )}
          </section>

          <section className="card aside-card missing-card">
            <div className="aside-eyebrow"><Info size={15} /> Telemetry limitations</div>
            <h3>Metadata boundaries</h3>
            <p>
              {incident.missing_evidence && incident.missing_evidence.length > 0
                ? `Missing fields: ${incident.missing_evidence.join(', ')}. Context required before attributing intent.`
                : 'Offline dataset lacks host identities and device telemetry. Human review is required.'}
            </p>
          </section>

          <section className="card aside-card next-card">
            <div className="aside-eyebrow"><Check size={15} /> What to check next</div>
            <ol>
              {incident.nextSteps.map((step, idx) => (
                <li key={idx}>{step}</li>
              ))}
            </ol>
          </section>
        </aside>
      </div>
    </>
  )
}

function EvidenceItem({ icon, title, copy, tone }: { icon: any; title: string; copy: string; tone: string }) {
  const Icon = resolveIcon(icon)
  return (
    <div className="evidence-item">
      <span className={`evidence-icon evidence-${tone}`}><Icon size={16} /></span>
      <div><strong>{title}</strong><p>{copy}</p></div>
    </div>
  )
}

function TimelineItem({ time, title, copy, tone }: TimelineEvent) {
  return (
    <div className="timeline-item">
      <span className="timeline-time">{time}</span>
      <span className={`timeline-node timeline-${tone}`} />
      <div><strong>{title}</strong><p>{copy}</p></div>
    </div>
  )
}

function EngineLine({ name, value, color }: { name: string; value: string; color: string }) {
  return (
    <div className="engine-line">
      <span><i className={`engine-dot engine-${color}`} />{name}</span>
      <strong>{value}</strong>
    </div>
  )
}

// ----------------------------------------------------------------------------
// 6. Insights Page
// ----------------------------------------------------------------------------

function Insights({
  data,
  loading,
}: {
  data: InsightsData | null
  loading: boolean
}) {
  const categories = data?.category_data || initialCategoryData
  const week = data?.week_data || initialWeekData
  const engines = data?.engine_data || initialEngineData

  return (
    <>
      <PageHeader
        eyebrow="Security insights"
        title="Understand your flow patterns"
        description="A thoughtful view of attack families, anomaly trends, detection-engine benchmarks, and candidate incident patterns."
      />

      <div className="insight-callout">
        <div className="insight-orb"><Lightbulb size={22} /></div>
        <div>
          <span className="eyebrow">A foundational principle</span>
          <h2>A baseline is the start, not a verdict.</h2>
          <p>
            Machine learning flow analysis highlights statistical deviations and attack signatures. It cannot replace business context or analyst verification.
          </p>
        </div>
        <div className="callout-status">
          <strong>Verified ML models</strong>
          <span>Phase 1–4 Benchmark Intelligence</span>
        </div>
      </div>

      <div className="content-grid insights-grid">
        <section className="card category-card">
          <CardHeading title="What VYUHA noticed" subtitle="Flow distribution by attack category" />
          <div className="donut-layout">
            <div className="donut-chart">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={categories} innerRadius={58} outerRadius={82} paddingAngle={4} dataKey="value" stroke="none">
                    {categories.map((item) => (
                      <Cell key={item.name} fill={item.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-center">
                <strong>{categories.length}</strong>
                <span>categories</span>
              </div>
            </div>
            <div className="category-legend">
              {categories.map((item) => (
                <div key={item.name}>
                  <span><i style={{ background: item.color }} />{item.name}</span>
                  <strong>{formatNumber(item.value)} flows</strong>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="card trend-card">
          <CardHeading title="Activity rhythm" subtitle="Observed normal vs suspicious flows" />
          <div className="chart-wrap chart-wrap-medium">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={week} margin={{ top: 12, right: 8, left: -22, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="#e7e1d7" strokeDasharray="4 5" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9b958c', fontSize: 11 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#9b958c', fontSize: 11 }} />
                <Tooltip content={<ChartTooltip />} />
                <Line type="monotone" dataKey="normal" name="Observed" stroke="#89a596" strokeWidth={2.5} dot={{ r: 3, fill: '#89a596', strokeWidth: 0 }} />
                <Line type="monotone" dataKey="unusual" name="Suspicious" stroke="#d47b62" strokeWidth={2.5} dot={{ r: 3, fill: '#d47b62', strokeWidth: 0 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <div className="content-grid insights-lower-grid">
        <section className="card">
          <CardHeading title="Detection engine benchmarks" subtitle="Empirical precision and macro F1 scores" />
          <div className="engine-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={engines} layout="vertical" margin={{ top: 0, right: 10, left: 20, bottom: 0 }} barGap={4}>
                <CartesianGrid horizontal={false} stroke="#e7e1d7" strokeDasharray="4 5" />
                <XAxis type="number" axisLine={false} tickLine={false} tick={{ fill: '#9b958c', fontSize: 11 }} domain={[0, 100]} />
                <YAxis type="category" dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#6b6862', fontSize: 11 }} width={160} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="reviewed" name="Baseline" fill="#e6e1d6" radius={[0, 5, 5, 0]} />
                <Bar dataKey="matched" name="Score (%)" fill="#89a596" radius={[0, 5, 5, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card summary-card">
          <CardHeading title="Explanatory guidance" subtitle="General operational principles based on model architecture" />
          <div className="summary-points">
            <SummaryPoint
              number="01"
              title="Supervised models provide attack clarity"
              copy={
                data?.benchmark_metrics?.binary_rf_precision
                  ? `Phase 3B Random Forest achieved ${(data.benchmark_metrics.binary_rf_precision * 100).toFixed(1)}% precision and ${(data.benchmark_metrics.binary_rf_recall * 100).toFixed(1)}% recall on known attack signatures.`
                  : 'Phase 3B and 3C Random Forests achieve high precision on known attack signatures in benchmark tests.'
              }
            />
            <SummaryPoint
              number="02"
              title="Unsupervised isolation flags novel anomalies"
              copy={
                data?.benchmark_metrics?.anomaly_iforest_roc_auc
                  ? `Phase 3D Isolation Forest achieved ${(data.benchmark_metrics.anomaly_iforest_roc_auc * 100).toFixed(1)}% ROC-AUC detecting distribution shifts without requiring attack labels.`
                  : 'Phase 3D Isolation Forest detects statistical distribution shifts without requiring attack labels.'
              }
            />
            <SummaryPoint
              number="03"
              title="Incident grouping reduces alert fatigue"
              copy="Phase 4 graph correlation connects corroborating alerts sharing destination ports and engine agreement into coherent candidate incidents."
            />
          </div>
        </section>
      </div>
    </>
  )
}

function SummaryPoint({ number, title, copy }: { number: string; title: string; copy: string }) {
  return (
    <div className="summary-point">
      <span>{number}</span>
      <div><strong>{title}</strong><p>{copy}</p></div>
    </div>
  )
}

export default App
