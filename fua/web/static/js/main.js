// FUA Web Interface JavaScript

// Global variables
let socket = null;
let statusCheckInterval = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize application
function initializeApp() {
    console.log('Initializing FUA Web Interface...');
    
    // Check system status
    checkSystemStatus();
    
    // Start status monitoring
    startStatusMonitoring();
    
    // Load quick stats
    loadQuickStats();
    
    // Initialize page-specific scripts
    initializePageScripts();
}

// Check system status
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        updateStatusIndicator(data.services);
    } catch (error) {
        console.error('Failed to check system status:', error);
        updateStatusIndicator({
            mlflow: false,
            distributed_monitor: false,
            experiment_tracker: false,
            model_registry: false
        });
    }
}

// Update status indicator
function updateStatusIndicator(services) {
    const indicator = document.getElementById('status-indicator');
    const text = document.getElementById('status-text');
    
    const allServicesUp = Object.values(services).every(service => service);
    
    if (allServicesUp) {
        indicator.className = 'fas fa-circle text-success';
        text.textContent = 'All Systems Operational';
    } else {
        indicator.className = 'fas fa-circle text-danger';
        text.textContent = 'Some Services Offline';
    }
}

// Start status monitoring
function startStatusMonitoring() {
    // Check status every 30 seconds
    statusCheckInterval = setInterval(checkSystemStatus, 30000);
}

// Load quick stats
async function loadQuickStats() {
    try {
        // Load experiment summary
        const summaryResponse = await fetch('/api/experiment/summary');
        const summary = await summaryResponse.json();
        
        if (summary.total_experiments !== undefined) {
            updateStat('total-experiments', summary.total_experiments);
        }
        
        if (summary.active_runs !== undefined) {
            updateStat('active-runs', summary.active_runs);
        }
        
        // Load models count
        const modelsResponse = await fetch('/api/models');
        const modelsData = await modelsResponse.json();
        
        if (modelsData.models) {
            updateStat('registered-models', modelsData.models.length);
        }
        
        // Update system health
        updateStat('system-health', '98%');
        
    } catch (error) {
        console.error('Failed to load quick stats:', error);
    }
}

// Update stat element
function updateStat(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = value;
        element.parentElement.classList.add('fade-in');
    }
}

// Initialize page-specific scripts
function initializePageScripts() {
    const path = window.location.pathname;
    
    switch (path) {
        case '/dashboard':
            initializeDashboard();
            break;
        case '/experiments':
            initializeExperimentsPage();
            break;
        case '/models':
            initializeModelsPage();
            break;
        case '/monitoring':
            initializeMonitoringPage();
            break;
    }
}

// Dashboard initialization
function initializeDashboard() {
    console.log('Initializing dashboard...');
    
    // Load dashboard data
    loadDashboardData();
    
    // Set up auto-refresh
    setInterval(loadDashboardData, 10000);
}

// Load dashboard data
async function loadDashboardData() {
    try {
        // Load experiment summary
        const summary = await fetch('/api/experiment/summary').then(r => r.json());
        updateDashboardSummary(summary);
        
        // Load recent experiments
        const experiments = await fetch('/api/experiments').then(r => r.json());
        updateRecentExperiments(experiments.experiments);
        
        // Load monitoring metrics
        const metrics = await fetch('/api/monitoring/metrics').then(r => r.json());
        updateMonitoringMetrics(metrics);
        
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
    }
}

// Update dashboard summary
function updateDashboardSummary(summary) {
    const container = document.getElementById('dashboard-summary');
    if (!container) return;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Total Runs</div>
                    <div class="metric-value text-primary">${summary.total_runs || 0}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Active Runs</div>
                    <div class="metric-value text-info">${summary.active_runs || 0}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Completed</div>
                    <div class="metric-value text-success">${summary.finished_runs || 0}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Failed</div>
                    <div class="metric-value text-danger">${summary.failed_runs || 0}</div>
                </div>
            </div>
        </div>
    `;
}

// Update recent experiments
function updateRecentExperiments(experiments) {
    const container = document.getElementById('recent-experiments');
    if (!container) return;
    
    const recentExperiments = experiments
        .sort((a, b) => new Date(b.start_time) - new Date(a.start_time))
        .slice(0, 5);
    
    container.innerHTML = recentExperiments.map(exp => `
        <tr>
            <td>${exp.run_name || 'N/A'}</td>
            <td><span class="badge bg-${exp.status === 'FINISHED' ? 'success' : 'warning'}">${exp.status}</span></td>
            <td>${exp.start_time ? new Date(exp.start_time).toLocaleString() : 'N/A'}</td>
            <td>${exp.metrics ? (exp.metrics.val_acc || exp.metrics.train_acc || 'N/A').toFixed(3) : 'N/A'}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="viewExperimentDetails('${exp.run_id}')">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

// Update monitoring metrics
function updateMonitoringMetrics(metrics) {
    const container = document.getElementById('monitoring-metrics');
    if (!container) return;
    
    if (metrics.aggregated) {
        const cpu = metrics.aggregated.cpu || {};
        const memory = metrics.aggregated.memory || {};
        
        container.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <div class="chart-container">
                        <h5>CPU Usage</h5>
                        <div class="progress" style="height: 30px;">
                            <div class="progress-bar" role="progressbar" style="width: ${cpu.avg || 0}%">
                                ${Math.round(cpu.avg || 0)}%
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="chart-container">
                        <h5>Memory Usage</h5>
                        <div class="progress" style="height: 30px;">
                            <div class="progress-bar bg-info" role="progressbar" style="width: ${memory.avg || 0}%">
                                ${Math.round(memory.avg || 0)}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
}

// Experiments page initialization
function initializeExperimentsPage() {
    console.log('Initializing experiments page...');
    loadExperiments();
}

// Load experiments
async function loadExperiments() {
    try {
        const response = await fetch('/api/experiments');
        const data = await response.json();
        
        const container = document.getElementById('experiments-table');
        if (!container) return;
        
        container.innerHTML = data.experiments.map(exp => `
            <tr>
                <td>${exp.run_id}</td>
                <td>${exp.run_name || 'N/A'}</td>
                <td><span class="badge bg-${getStatusColor(exp.status)}">${exp.status}</span></td>
                <td>${exp.start_time ? new Date(exp.start_time).toLocaleString() : 'N/A'}</td>
                <td>${exp.end_time ? new Date(exp.end_time).toLocaleString() : 'N/A'}</td>
                <td>
                    <button class="btn btn-sm btn-primary" onclick="viewExperimentDetails('${exp.run_id}')">
                        View Details
                    </button>
                </td>
            </tr>
        `).join('');
        
    } catch (error) {
        console.error('Failed to load experiments:', error);
    }
}

// Get status color
function getStatusColor(status) {
    switch (status) {
        case 'FINISHED': return 'success';
        case 'RUNNING': return 'primary';
        case 'FAILED': return 'danger';
        default: return 'secondary';
    }
}

// View experiment details
async function viewExperimentDetails(runId) {
    try {
        const response = await fetch(`/api/experiments/${runId}`);
        const experiment = await response.json();
        
        // Show modal with experiment details
        showModal('Experiment Details', formatExperimentDetails(experiment));
        
    } catch (error) {
        console.error('Failed to load experiment details:', error);
        showAlert('Failed to load experiment details', 'danger');
    }
}

// Format experiment details
function formatExperimentDetails(experiment) {
    return `
        <div class="row">
            <div class="col-md-6">
                <h6>Run Information</h6>
                <p><strong>Run ID:</strong> ${experiment.run_id}</p>
                <p><strong>Name:</strong> ${experiment.run_name || 'N/A'}</p>
                <p><strong>Status:</strong> ${experiment.status}</p>
                <p><strong>Start Time:</strong> ${experiment.start_time ? new Date(experiment.start_time).toLocaleString() : 'N/A'}</p>
                <p><strong>End Time:</strong> ${experiment.end_time ? new Date(experiment.end_time).toLocaleString() : 'N/A'}</p>
            </div>
            <div class="col-md-6">
                <h6>Metrics</h6>
                ${formatMetrics(experiment.metrics)}
            </div>
        </div>
    `;
}

// Format metrics
function formatMetrics(metrics) {
    if (!metrics) return '<p>No metrics available</p>';
    
    return Object.entries(metrics).map(([key, value]) => `
        <p><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(4) : value}</p>
    `).join('');
}

// Models page initialization
function initializeModelsPage() {
    console.log('Initializing models page...');
    loadModels();
}

// Load models
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const container = document.getElementById('models-container');
        if (!container) return;
        
        container.innerHTML = data.models.map(model => `
            <div class="model-card">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h5>${model.name}</h5>
                        <p class="text-muted">${model.description || 'No description'}</p>
                        ${model.latest_version ? `
                            <div class="mt-2">
                                <span class="model-version">v${model.latest_version.version}</span>
                                <span class="model-stage ${model.latest_version.stage}">${model.latest_version.stage}</span>
                            </div>
                        ` : ''}
                    </div>
                    <div>
                        <button class="btn btn-sm btn-outline-primary me-2" onclick="viewModelDetails('${model.name}')">
                            <i class="fas fa-info-circle"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-success" onclick="deployModel('${model.name}')">
                            <i class="fas fa-rocket"></i>
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

// Monitoring page initialization
function initializeMonitoringPage() {
    console.log('Initializing monitoring page...');
    loadMonitoringData();
    
    // Set up real-time updates
    setInterval(loadMonitoringData, 5000);
}

// Load monitoring data
async function loadMonitoringData() {
    try {
        const response = await fetch('/api/monitoring/metrics');
        const data = await response.json();
        
        updateMonitoringCharts(data);
        
    } catch (error) {
        console.error('Failed to load monitoring data:', error);
    }
}

// Update monitoring charts
function updateMonitoringCharts(metrics) {
    // Update CPU chart
    updateLineChart('cpu-chart', metrics.cpu_history || []);
    
    // Update memory chart
    updateLineChart('memory-chart', metrics.memory_history || []);
    
    // Update latency chart
    updateLineChart('latency-chart', metrics.latency_history || []);
}

// Update line chart
function updateLineChart(chartId, data) {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;
    
    // This is a placeholder - in a real implementation, you would use Chart.js or similar
    console.log(`Updating chart ${chartId} with data:`, data);
}

// Utility functions
function showModal(title, content) {
    const modalHtml = `
        <div class="modal fade" id="detailModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">${title}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        ${content}
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal
    const existingModal = document.getElementById('detailModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Add new modal
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('detailModal'));
    modal.show();
}

function showAlert(message, type = 'info') {
    const alertHtml = `
        <div class="alert alert-${type} alert-custom alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    let container = document.querySelector('.alert-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'alert-container';
        document.body.appendChild(container);
    }
    
    container.insertAdjacentHTML('beforeend', alertHtml);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = container.querySelector('.alert');
        if (alert) {
            alert.remove();
        }
    }, 5000);
}

// Export functions for global access
window.viewExperimentDetails = viewExperimentDetails;
window.viewModelDetails = viewModelDetails;
window.deployModel = deployModel;