/**
 * Retail Brain — Frontend Application Logic
 * Interacts with FastAPI backend for Auth, Predictions, and Real-time Alerts.
 */

const API_BASE = "http://localhost:8000";
const WS_BASE = "ws://localhost:8000";

const state = {
    token: localStorage.getItem("rb_token") || null,
    user: null,
    view: "overview"
};

// ── DOM Elements ───────────────────────────────────────────────────
const authView = document.getElementById("auth-view");
const dashboardView = document.getElementById("dashboard-view");
const loginForm = document.getElementById("login-form");
const loginError = document.getElementById("login-error");
const navItems = document.querySelectorAll(".nav-item");
const sections = document.querySelectorAll(".dashboard-section");

const riskTbody = document.getElementById("risk-tbody");
const auditTbody = document.getElementById("audit-tbody");
const alertContainer = document.getElementById("alert-container");

const displayEmail = document.getElementById("display-email");
const countCritical = document.getElementById("count-critical");

// ── Authentication ──────────────────────────────────────────────────

async function login(email, password) {
    loginError.textContent = "";
    try {
        const formData = new URLSearchParams();
        formData.append("username", email);
        formData.append("password", password);

        const response = await fetch(`${API_BASE}/api/auth/token`, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: formData
        });

        if (!response.ok) throw new Error("Invalid credentials or server error.");

        const data = await response.json();
        state.token = data.access_token;
        localStorage.setItem("rb_token", state.token);
        
        await initDashboard();
    } catch (err) {
        loginError.textContent = err.message;
    }
}

async function fetchMe() {
    if (!state.token) return null;
    try {
        const res = await fetch(`${API_BASE}/api/auth/me`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        if (!res.ok) throw new Error("Session expired");
        return await res.json();
    } catch {
        logout();
        return null;
    }
}

function logout() {
    state.token = null;
    state.user = null;
    localStorage.removeItem("rb_token");
    showView("auth");
}

// ── Data Fetching ───────────────────────────────────────────────────

async function refreshRiskData() {
    try {
        const res = await fetch(`${API_BASE}/api/predictions/risk`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        const data = await res.json();
        renderRiskTable(data);
        
        const criticals = data.filter(p => p.stockout_probability >= 0.8).length;
        countCritical.textContent = criticals;
    } catch (err) {
        console.error("Failed to fetch risk data", err);
    }
}

async function refreshAuditLogs() {
    try {
        const res = await fetch(`${API_BASE}/api/enterprise/audit?limit=20`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        const data = await res.json();
        renderAuditTable(data);
    } catch (err) {
        console.error("Failed to fetch audit logs", err);
    }
}

// ── Rendering ───────────────────────────────────────────────────────

function renderRiskTable(products) {
    riskTbody.innerHTML = "";
    products.forEach(p => {
        const row = document.createElement("tr");
        const prob = (p.stockout_probability * 100).toFixed(0);
        const riskClass = p.stockout_probability >= 0.8 ? "risk-high" : "risk-low";
        
        row.innerHTML = `
            <td>
                <div style="font-weight: 600;">${p.product_name}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">SKU: ${p.product_id}</div>
            </td>
            <td>${p.category}</td>
            <td>${p.stock_on_hand}</td>
            <td>${p.days_of_cover}d</td>
            <td><span class="risk-chip ${riskClass}">${prob}%</span></td>
            <td>${p.stockout_predicted ? '<i class="ph-fill ph-warning-circle" style="color:var(--danger)"></i> Stockout' : '<i class="ph-fill ph-check-circle" style="color:var(--success)"></i> Optimal'}</td>
        `;
        riskTbody.appendChild(row);
    });
}

function renderAuditTable(logs) {
    auditTbody.innerHTML = "";
    logs.forEach(log => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${new Date(log.timestamp).toLocaleString()}</td>
            <td><code style="color:var(--primary)">${log.user_id || 'system'}</code></td>
            <td><span style="font-weight:600">${log.action.toUpperCase()}</span></td>
            <td style="color:var(--text-muted); font-size:0.85rem">${log.details}</td>
        `;
        auditTbody.appendChild(row);
    });
}

// ── Real-time Alerts (WebSockets) ──────────────────────────────────

function initWebSockets() {
    const storeId = state.user.store_id || "ALL";
    const socket = new WebSocket(`${WS_BASE}/api/ws/alerts/${storeId}`);

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        showToast(data.message || "New Inventory Alert!");
        refreshRiskData(); // Live refresh
    };

    socket.onclose = () => {
        console.warn("WS disconnected. Retrying in 5s...");
        setTimeout(initWebSockets, 5000);
    };
}

function showToast(message) {
    const toast = document.createElement("div");
    toast.className = "toast glass glowing-border";
    toast.innerHTML = `
        <i class="ph-fill ph-bell-ringing" style="color:var(--primary); font-size:1.5rem"></i>
        <div>
            <div style="font-weight:700; margin-bottom:0.25rem">Live Alert</div>
            <div style="font-size:0.85rem; color:var(--text-muted)">${message}</div>
        </div>
    `;
    alertContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 6000);
}

// ── UI Control ──────────────────────────────────────────────────────

function showView(view) {
    if (view === "auth") {
        authView.classList.remove("hidden");
        dashboardView.classList.add("hidden");
    } else {
        authView.classList.add("hidden");
        dashboardView.classList.remove("hidden");
    }
}

async function initDashboard() {
    const user = await fetchMe();
    if (!user) return;
    
    state.user = user;
    displayEmail.textContent = user.email;
    document.getElementById("display-store-id").textContent = `ID: ${user.store_id || 'GLOBAL'}`;
    
    showView("dashboard");
    refreshRiskData();
    initWebSockets();
}

// ── Event Listeners ────────────────────────────────────────────────

loginForm.addEventListener("submit", (e) => {
    e.preventDefault();
    login(document.getElementById("email").value, document.getElementById("password").value);
});

document.getElementById("logout-btn").addEventListener("click", (e) => {
    e.preventDefault();
    logout();
});

navItems.forEach(item => {
    item.addEventListener("click", (e) => {
        e.preventDefault();
        const view = item.getAttribute("data-view");
        if (!view) return;

        navItems.forEach(i => i.classList.remove("active"));
        item.classList.add("active");

        sections.forEach(s => s.classList.add("hidden"));
        document.getElementById(`section-${view}`).classList.remove("hidden");

        if (view === "audit") refreshAuditLogs();
        if (view === "overview") refreshRiskData();
    });
});

document.getElementById("export-csv").addEventListener("click", async () => {
    try {
        const response = await fetch(`${API_BASE}/api/enterprise/reports/risk/csv`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `stockout_risk_report_${new Date().toISOString().slice(0,10)}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        showToast("Risk report exported successfully.");
    } catch (err) {
        showToast("Export failed.");
    }
});

// ── Initialize ─────────────────────────────────────────────────────

if (state.token) {
    initDashboard();
} else {
    showView("auth");
}
