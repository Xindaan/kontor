"""
Health and Integrity Reports for constituent data.

Provides diagnostics on data quality, coverage, and potential issues.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

from .base import ConstituentProvider
from .models import (
    ConstituentSnapshot,
    DataQuality,
    IdentifierType,
    MemberIdentifier,
)

logger = logging.getLogger(__name__)


@dataclass
class IntegrityIssue:
    """A detected integrity issue."""

    severity: str  # "error", "warning", "info"
    category: str  # "coverage", "count", "identifier", "gap"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class HealthReport:
    """
    Comprehensive health report for a constituent provider.

    Contains coverage information, integrity checks, and diagnostics.
    """

    index_id: str
    provider_id: str
    quality: DataQuality
    generated_at: date

    # Coverage
    coverage_start: Optional[date] = None
    coverage_end: Optional[date] = None
    total_snapshots: int = 0
    frequency: str = "unknown"  # "daily", "monthly", "event-based"

    # Member statistics
    avg_member_count: float = 0.0
    min_member_count: int = 0
    max_member_count: int = 0
    member_count_stddev: float = 0.0

    # Identifier quality
    identifier_breakdown: Dict[str, int] = field(default_factory=dict)
    pct_with_stable_id: float = 0.0  # CUSIP/ISIN/SEDOL

    # Issues
    issues: List[IntegrityIssue] = field(default_factory=list)

    # Raw data for analysis
    snapshot_sizes: List[Tuple[date, int]] = field(default_factory=list)
    gaps: List[Tuple[date, date]] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    @property
    def is_healthy(self) -> bool:
        return not self.has_errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_id": self.index_id,
            "provider_id": self.provider_id,
            "quality": self.quality.value,
            "generated_at": self.generated_at.isoformat(),
            "coverage": {
                "start": self.coverage_start.isoformat() if self.coverage_start else None,
                "end": self.coverage_end.isoformat() if self.coverage_end else None,
                "total_snapshots": self.total_snapshots,
                "frequency": self.frequency,
            },
            "members": {
                "avg_count": round(self.avg_member_count, 1),
                "min_count": self.min_member_count,
                "max_count": self.max_member_count,
                "stddev": round(self.member_count_stddev, 2),
            },
            "identifiers": {
                "breakdown": self.identifier_breakdown,
                "pct_with_stable_id": round(self.pct_with_stable_id, 1),
            },
            "issues": [i.to_dict() for i in self.issues],
            "summary": {
                "healthy": self.is_healthy,
                "errors": sum(1 for i in self.issues if i.severity == "error"),
                "warnings": sum(1 for i in self.issues if i.severity == "warning"),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_report(self) -> None:
        """Print human-readable report to console."""
        print(f"\n{'='*60}")
        print(f"Health Report: {self.index_id}")
        print(f"{'='*60}")
        print(f"Provider: {self.provider_id}")
        print(f"Quality: {self.quality.value}")
        print(f"Generated: {self.generated_at}")
        print()

        print("COVERAGE")
        print("-" * 40)
        print(f"  Start: {self.coverage_start or 'N/A'}")
        print(f"  End: {self.coverage_end or 'N/A'}")
        print(f"  Snapshots: {self.total_snapshots}")
        print(f"  Frequency: {self.frequency}")
        print()

        print("MEMBER STATISTICS")
        print("-" * 40)
        print(f"  Average count: {self.avg_member_count:.1f}")
        print(f"  Min/Max: {self.min_member_count} / {self.max_member_count}")
        print(f"  Std dev: {self.member_count_stddev:.2f}")
        print()

        print("IDENTIFIER QUALITY")
        print("-" * 40)
        for id_type, count in self.identifier_breakdown.items():
            print(f"  {id_type}: {count}")
        print(f"  Stable ID coverage: {self.pct_with_stable_id:.1f}%")
        print()

        if self.issues:
            print("ISSUES")
            print("-" * 40)
            for issue in self.issues:
                icon = {"error": "[!]", "warning": "[!]", "info": "[i]"}[issue.severity]
                print(f"  {icon} [{issue.severity.upper()}] {issue.message}")
        else:
            print("No issues detected.")

        print()
        status = "HEALTHY" if self.is_healthy else "UNHEALTHY"
        print(f"Status: {status}")
        print(f"{'='*60}\n")

    @classmethod
    def generate(
        cls,
        provider: ConstituentProvider,
        sample_dates: Optional[List[date]] = None,
    ) -> "HealthReport":
        """
        Generate health report for a provider.

        Args:
            provider: The constituent provider to analyze
            sample_dates: Optional list of dates to sample (default: all available)

        Returns:
            HealthReport with analysis results
        """
        report = cls(
            index_id=provider.index_id,
            provider_id=provider.id,
            quality=provider.quality,
            generated_at=date.today(),
        )

        try:
            available = provider.available_dates()

            if not available:
                report.issues.append(
                    IntegrityIssue(
                        severity="error",
                        category="coverage",
                        message="No data available",
                    )
                )
                return report

            # Coverage
            report.coverage_start = min(available)
            report.coverage_end = max(available)
            report.total_snapshots = len(available)

            # Detect frequency
            report.frequency = _detect_frequency(available)

            # Sample snapshots for analysis
            if sample_dates:
                dates_to_check = [d for d in sample_dates if d in available]
            else:
                # Sample up to 100 dates evenly distributed
                if len(available) <= 100:
                    dates_to_check = available
                else:
                    step = len(available) // 100
                    dates_to_check = available[::step]

            # Analyze snapshots
            sizes = []
            id_types: Counter = Counter()
            stable_id_count = 0
            total_members = 0

            for d in dates_to_check:
                try:
                    snapshot = provider.snapshot(d)
                    sizes.append((d, snapshot.size))
                    report.snapshot_sizes.append((d, snapshot.size))

                    for member in snapshot.members:
                        id_types[member.id_type.value] += 1
                        total_members += 1

                        if member.id_type in (
                            IdentifierType.CUSIP,
                            IdentifierType.ISIN,
                            IdentifierType.SEDOL,
                        ):
                            stable_id_count += 1

                except Exception as e:
                    logger.warning(f"Failed to get snapshot for {d}: {e}")

            # Calculate statistics
            if sizes:
                counts = [s[1] for s in sizes]
                report.avg_member_count = sum(counts) / len(counts)
                report.min_member_count = min(counts)
                report.max_member_count = max(counts)

                if len(counts) > 1:
                    mean = report.avg_member_count
                    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
                    report.member_count_stddev = variance ** 0.5

            # Identifier breakdown
            report.identifier_breakdown = dict(id_types)
            if total_members > 0:
                report.pct_with_stable_id = (stable_id_count / total_members) * 100

            # Detect gaps
            report.gaps = _detect_gaps(available, report.frequency)

            # Check for issues
            _check_issues(report, provider)

        except Exception as e:
            logger.error(f"Failed to generate health report: {e}")
            report.issues.append(
                IntegrityIssue(
                    severity="error",
                    category="analysis",
                    message=f"Analysis failed: {str(e)}",
                )
            )

        return report


class IntegrityCheck:
    """
    Run integrity checks on constituent data.

    Validates data quality and consistency.
    """

    @staticmethod
    def check_all(provider: ConstituentProvider) -> List[IntegrityIssue]:
        """Run all integrity checks."""
        issues = []

        issues.extend(IntegrityCheck.check_coverage(provider))
        issues.extend(IntegrityCheck.check_member_counts(provider))
        issues.extend(IntegrityCheck.check_identifier_quality(provider))

        return issues

    @staticmethod
    def check_coverage(provider: ConstituentProvider) -> List[IntegrityIssue]:
        """Check data coverage."""
        issues = []

        try:
            available = provider.available_dates()

            if not available:
                issues.append(
                    IntegrityIssue(
                        severity="error",
                        category="coverage",
                        message="No data available",
                    )
                )
                return issues

            start = min(available)
            end = max(available)

            # Check if data is stale
            if (date.today() - end).days > 90:
                issues.append(
                    IntegrityIssue(
                        severity="warning",
                        category="coverage",
                        message=f"Data may be stale (last snapshot: {end})",
                        details={"last_date": end.isoformat()},
                    )
                )

            # Check for large gaps
            frequency = _detect_frequency(available)
            gaps = _detect_gaps(available, frequency)

            for gap_start, gap_end in gaps:
                gap_days = (gap_end - gap_start).days
                if gap_days > 90:
                    issues.append(
                        IntegrityIssue(
                            severity="warning",
                            category="gap",
                            message=f"Large data gap: {gap_start} to {gap_end} ({gap_days} days)",
                            details={
                                "start": gap_start.isoformat(),
                                "end": gap_end.isoformat(),
                                "days": gap_days,
                            },
                        )
                    )

        except Exception as e:
            issues.append(
                IntegrityIssue(
                    severity="error",
                    category="coverage",
                    message=f"Coverage check failed: {str(e)}",
                )
            )

        return issues

    @staticmethod
    def check_member_counts(provider: ConstituentProvider) -> List[IntegrityIssue]:
        """Check member count consistency."""
        issues = []

        try:
            available = provider.available_dates()
            if not available:
                return issues

            # Sample some dates
            sample = available[::max(1, len(available) // 20)]

            counts = []
            for d in sample:
                snapshot = provider.snapshot(d)
                counts.append((d, snapshot.size))

            if counts:
                sizes = [c[1] for c in counts]
                avg = sum(sizes) / len(sizes)

                # Check for dramatic changes
                for d, size in counts:
                    if avg > 0:
                        deviation = abs(size - avg) / avg
                        if deviation > 0.2:  # More than 20% deviation
                            issues.append(
                                IntegrityIssue(
                                    severity="warning",
                                    category="count",
                                    message=f"Unusual member count on {d}: {size} (avg: {avg:.0f})",
                                    details={
                                        "date": d.isoformat(),
                                        "count": size,
                                        "average": round(avg),
                                        "deviation_pct": round(deviation * 100, 1),
                                    },
                                )
                            )

                # Check for empty snapshots
                empty = [d for d, size in counts if size == 0]
                if empty:
                    issues.append(
                        IntegrityIssue(
                            severity="error",
                            category="count",
                            message=f"Empty snapshots found: {len(empty)} dates",
                            details={"empty_dates": [d.isoformat() for d in empty[:5]]},
                        )
                    )

        except Exception as e:
            issues.append(
                IntegrityIssue(
                    severity="error",
                    category="count",
                    message=f"Count check failed: {str(e)}",
                )
            )

        return issues

    @staticmethod
    def check_identifier_quality(provider: ConstituentProvider) -> List[IntegrityIssue]:
        """Check identifier quality."""
        issues = []

        try:
            available = provider.available_dates()
            if not available:
                return issues

            # Sample recent snapshot
            recent = max(available)
            snapshot = provider.snapshot(recent)

            # Count identifier types
            id_counts: Counter = Counter()
            name_only = 0

            for member in snapshot.members:
                id_counts[member.id_type.value] += 1
                if member.id_type == IdentifierType.NAME:
                    name_only += 1

            # Check for name-only identifiers
            if snapshot.size > 0:
                name_pct = (name_only / snapshot.size) * 100
                if name_pct > 20:
                    issues.append(
                        IntegrityIssue(
                            severity="warning",
                            category="identifier",
                            message=f"High percentage of name-only identifiers: {name_pct:.1f}%",
                            details={
                                "name_only_count": name_only,
                                "total": snapshot.size,
                            },
                        )
                    )

            # Check for no stable IDs
            stable = sum(
                id_counts.get(t, 0)
                for t in ["cusip", "isin", "sedol"]
            )
            if stable == 0 and snapshot.size > 0:
                issues.append(
                    IntegrityIssue(
                        severity="info",
                        category="identifier",
                        message="No stable identifiers (CUSIP/ISIN/SEDOL) available",
                        details={"identifier_types": dict(id_counts)},
                    )
                )

        except Exception as e:
            issues.append(
                IntegrityIssue(
                    severity="error",
                    category="identifier",
                    message=f"Identifier check failed: {str(e)}",
                )
            )

        return issues


def _detect_frequency(dates: List[date]) -> str:
    """Detect the frequency of snapshots."""
    if len(dates) < 2:
        return "unknown"

    # Calculate average gap
    gaps = []
    sorted_dates = sorted(dates)
    for i in range(1, len(sorted_dates)):
        gap = (sorted_dates[i] - sorted_dates[i - 1]).days
        gaps.append(gap)

    avg_gap = sum(gaps) / len(gaps)

    if avg_gap <= 2:
        return "daily"
    elif avg_gap <= 8:
        return "weekly"
    elif avg_gap <= 35:
        return "monthly"
    elif avg_gap <= 100:
        return "quarterly"
    else:
        return "event-based"


def _detect_gaps(dates: List[date], frequency: str) -> List[Tuple[date, date]]:
    """Detect significant gaps in data."""
    if len(dates) < 2:
        return []

    # Expected gap based on frequency
    expected_gaps = {
        "daily": 5,
        "weekly": 14,
        "monthly": 45,
        "quarterly": 120,
        "event-based": 180,
        "unknown": 90,
    }

    max_expected = expected_gaps.get(frequency, 90)
    gaps = []

    sorted_dates = sorted(dates)
    for i in range(1, len(sorted_dates)):
        gap = (sorted_dates[i] - sorted_dates[i - 1]).days
        if gap > max_expected:
            gaps.append((sorted_dates[i - 1], sorted_dates[i]))

    return gaps


def _check_issues(report: HealthReport, provider: ConstituentProvider) -> None:
    """Add issues to report based on analysis."""
    # Coverage issues
    if report.total_snapshots == 0:
        report.issues.append(
            IntegrityIssue(
                severity="error",
                category="coverage",
                message="No snapshots available",
            )
        )
        return

    # Stale data
    if report.coverage_end:
        days_old = (date.today() - report.coverage_end).days
        if days_old > 90:
            report.issues.append(
                IntegrityIssue(
                    severity="warning",
                    category="coverage",
                    message=f"Data may be stale ({days_old} days old)",
                    details={"last_date": report.coverage_end.isoformat()},
                )
            )

    # Large count variations
    if report.avg_member_count > 0:
        cv = report.member_count_stddev / report.avg_member_count
        if cv > 0.1:  # Coefficient of variation > 10%
            report.issues.append(
                IntegrityIssue(
                    severity="info",
                    category="count",
                    message=f"Member count varies significantly (CV: {cv:.1%})",
                    details={
                        "min": report.min_member_count,
                        "max": report.max_member_count,
                        "avg": round(report.avg_member_count),
                    },
                )
            )

    # Gaps
    for gap_start, gap_end in report.gaps:
        gap_days = (gap_end - gap_start).days
        if gap_days > 60:
            report.issues.append(
                IntegrityIssue(
                    severity="warning",
                    category="gap",
                    message=f"Data gap: {gap_start} to {gap_end}",
                    details={"days": gap_days},
                )
            )

    # Identifier quality
    if report.pct_with_stable_id < 50:
        report.issues.append(
            IntegrityIssue(
                severity="info",
                category="identifier",
                message=f"Low stable identifier coverage ({report.pct_with_stable_id:.1f}%)",
            )
        )

    # Quality-specific notes
    if provider.quality == DataQuality.PROXY_ETF_HOLDINGS:
        report.issues.append(
            IntegrityIssue(
                severity="info",
                category="quality",
                message="ETF holdings proxy - may differ from actual index",
            )
        )
    elif provider.quality == DataQuality.COMMUNITY_CHANGELOG:
        report.issues.append(
            IntegrityIssue(
                severity="info",
                category="quality",
                message="Community data - not audit-grade",
            )
        )
