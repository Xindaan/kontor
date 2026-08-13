/* Kontor Web UI - Shared Utilities */

/* ─── Number Formatting ─── */

/**
 * Format a number as currency (EUR, de-DE locale)
 */
function formatCurrency(value) {
    if (value === null || value === undefined) return '-';
    return new Intl.NumberFormat('de-DE', {
        style: 'currency',
        currency: 'EUR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

/**
 * Format a decimal as percentage (e.g., 0.1234 -> "12.34%")
 */
function formatPercent(value) {
    if (value === null || value === undefined) return '-';
    return (value * 100).toFixed(2) + '%';
}

/**
 * Format a decimal as signed percentage (e.g., 0.05 -> "+5.00%")
 */
function formatPercentSigned(value) {
    if (value === null || value === undefined) return '-';
    const pct = (value * 100).toFixed(2);
    return (value >= 0 ? '+' : '') + pct + '%';
}

/**
 * Format a number with 2 decimal places
 */
function formatNumber(value) {
    if (value === null || value === undefined) return '-';
    return Number(value).toFixed(2);
}

/**
 * Format seconds as human-readable duration (e.g., "2m 15s")
 */
function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins > 0) {
        return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
}

/**
 * Format a money value with sign (e.g., +1,234.56)
 */
function formatMoneySigned(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
    const num = Number(value);
    const sign = num > 0 ? '+' : '';
    return sign + formatMoney(num);
}

/**
 * Format a money value (without currency symbol, locale-aware)
 */
function formatMoney(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
    return Number(value).toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

/* ─── Key/Value Parsing ─── */

/**
 * Parse a text input like "equity=0.6, bonds=0.4" into an object.
 * If numericValues is true, values are parsed as numbers.
 */
function parseKeyValueMap(input, numericValues) {
    numericValues = numericValues || false;
    var text = (input || '').trim();
    if (!text) return {};
    var entries = text
        .split(/[\n,]+/)
        .map(function(s) { return s.trim(); })
        .filter(Boolean);
    var result = {};
    for (var i = 0; i < entries.length; i++) {
        var entry = entries[i];
        var separator = entry.includes('=') ? '=' : (entry.includes(':') ? ':' : null);
        if (!separator) {
            throw new Error('Invalid key/value entry: "' + entry + '". Use key=value.');
        }
        var parts = entry.split(separator, 2).map(function(s) { return s.trim(); });
        var rawKey = parts[0];
        var rawValue = parts[1];
        if (!rawKey || !rawValue) {
            throw new Error('Invalid key/value entry: "' + entry + '".');
        }
        if (numericValues) {
            var parsed = Number(rawValue);
            if (!Number.isFinite(parsed)) {
                throw new Error('Invalid numeric value in "' + entry + '".');
            }
            result[rawKey.toLowerCase()] = parsed;
        } else {
            result[rawKey] = rawValue.toLowerCase();
        }
    }
    return result;
}

/* ─── Plotly Theme ─── */

/**
 * Shared Plotly layout config for consistent chart styling.
 * Merge with your chart-specific layout using Object.assign or spread.
 */
var PLOTLY_LAYOUT = {
    font: {
        family: 'Inter, system-ui, sans-serif',
        color: '#475569',
        size: 12
    },
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    xaxis: {
        gridcolor: '#f1f5f9',
        zerolinecolor: '#e2e8f0',
        linecolor: '#e2e8f0'
    },
    yaxis: {
        gridcolor: '#f1f5f9',
        zerolinecolor: '#e2e8f0',
        linecolor: '#e2e8f0'
    },
    margin: { t: 20, r: 20, b: 40, l: 60 },
    hovermode: 'x unified',
    hoverlabel: {
        bgcolor: '#1e293b',
        font: { color: '#f8fafc', family: 'Inter, sans-serif', size: 12 }
    }
};

/**
 * Curated color palette for multi-series charts.
 */
var CHART_COLORS = [
    '#2563eb',  // blue
    '#16a34a',  // green
    '#d97706',  // amber
    '#dc2626',  // red
    '#7c3aed',  // purple
    '#0891b2',  // cyan
    '#db2777',  // pink
    '#ea580c'   // orange
];

/**
 * Get a deep-merged Plotly layout with the shared theme.
 */
function getPlotlyLayout(overrides) {
    overrides = overrides || {};
    var layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUT));
    // Merge top-level keys
    for (var key in overrides) {
        if (overrides.hasOwnProperty(key)) {
            if (typeof overrides[key] === 'object' && overrides[key] !== null && !Array.isArray(overrides[key]) &&
                typeof layout[key] === 'object' && layout[key] !== null) {
                for (var subKey in overrides[key]) {
                    if (overrides[key].hasOwnProperty(subKey)) {
                        layout[key][subKey] = overrides[key][subKey];
                    }
                }
            } else {
                layout[key] = overrides[key];
            }
        }
    }
    return layout;
}

/* ─── Timer Mixin (for Alpine.js components) ─── */

/**
 * Create a time-based progress tracker.
 */
function createTimeProgressTracker(options) {
    var intervalId = null;

    var update = function() {
        var elapsedSeconds = options.getElapsedSeconds();
        var totalSeconds = Math.max(options.estimateTotalSeconds(), 1);
        var percent = Math.min(99, (elapsedSeconds / totalSeconds) * 100);
        var remainingSeconds = Math.max(0, totalSeconds - elapsedSeconds);
        var message = options.getMessage(elapsedSeconds, totalSeconds);
        options.onUpdate({
            percent: percent,
            message: message,
            remainingSeconds: remainingSeconds,
            totalSeconds: totalSeconds
        });
    };

    return {
        start: function() {
            update();
            intervalId = setInterval(update, options.intervalMs || 200);
        },
        stop: function() {
            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
        }
    };
}

/* ─── Scroll Shadow (wide table indicator) ─── */

/**
 * Init scroll shadow indicators on all .scroll-shadow elements.
 * Adds/removes .scroll-shadow--right class based on scroll position.
 */
function initScrollShadows() {
    document.querySelectorAll('.scroll-shadow').forEach(function(el) {
        var check = function() {
            var hasOverflow = el.scrollWidth > el.clientWidth;
            var atEnd = el.scrollLeft + el.clientWidth >= el.scrollWidth - 2;
            el.classList.toggle('scroll-shadow--right', hasOverflow && !atEnd);
        };
        el.addEventListener('scroll', check);
        check();
        // Re-check after layout settles
        if (typeof ResizeObserver !== 'undefined') {
            new ResizeObserver(check).observe(el);
        }
    });
}

document.addEventListener('DOMContentLoaded', initScrollShadows);
// Also handle HTMX swaps
document.addEventListener('htmx:afterSwap', initScrollShadows);
