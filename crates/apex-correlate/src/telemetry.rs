//! The in-memory [`Telemetry`] container and its [`GridKind`].

use std::collections::BTreeMap;

use apex_telemetry::ChannelId;

/// Which coordinate the samples are (meant to be) uniform in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridKind {
    /// Arc length `s` (metres) — the [`ChannelId::S`] axis.
    S,
    /// Elapsed time `t` (seconds) — the [`ChannelId::Time`] axis.
    T,
}

impl GridKind {
    /// The registry channel that carries this grid's axis coordinate.
    pub fn axis_channel(self) -> ChannelId {
        match self {
            GridKind::S => ChannelId::S,
            GridKind::T => ChannelId::Time,
        }
    }

    /// The `# grid:` token (`"s"` / `"t"`).
    pub fn as_str(self) -> &'static str {
        match self {
            GridKind::S => "s",
            GridKind::T => "t",
        }
    }

    /// Parse a `# grid:` token.
    pub fn parse(s: &str) -> Option<GridKind> {
        match s.trim() {
            "s" => Some(GridKind::S),
            "t" => Some(GridKind::T),
            _ => None,
        }
    }
}

/// A single imported lap: a set of registry channels sampled on a common grid,
/// plus the free-form provenance metadata read from the file's comment header.
///
/// Every channel vector has the same length ([`Telemetry::len`]). The grid axis
/// channel ([`GridKind::axis_channel`]) is itself one of the channels.
///
/// # Missing data
///
/// Measured telemetry has real gaps. Non-finite (`NaN`) samples are **kept** —
/// a dropout is data, not an error — and are surfaced via
/// [`Telemetry::validity_mask`] / [`Telemetry::nan_count`] rather than being
/// silently dropped or interpolated away.
#[derive(Debug, Clone)]
pub struct Telemetry {
    /// The resampling / interpolation axis.
    pub grid: GridKind,
    /// Channel samples, keyed by [`ChannelId`] (ordered by registry declaration
    /// order, so column ordering is deterministic).
    pub channels: BTreeMap<ChannelId, Vec<f64>>,
    /// Free-form `# key: value` provenance lines, in file order (the reserved
    /// `grid` and `columns` keys are not included here).
    pub metadata: Vec<(String, String)>,
}

impl Telemetry {
    /// Number of samples (rows). Zero if there are no channels.
    pub fn len(&self) -> usize {
        self.channels.values().next().map_or(0, Vec::len)
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The grid axis samples (`s` or `t`), if present.
    pub fn axis(&self) -> Option<&[f64]> {
        self.channel(self.grid.axis_channel())
    }

    /// Samples for one channel, if present.
    pub fn channel(&self, id: ChannelId) -> Option<&[f64]> {
        self.channels.get(&id).map(Vec::as_slice)
    }

    /// Per-sample validity mask for a channel (`true` = finite), if present.
    pub fn validity_mask(&self, id: ChannelId) -> Option<Vec<bool>> {
        self.channels
            .get(&id)
            .map(|v| v.iter().map(|x| x.is_finite()).collect())
    }

    /// Count of non-finite (`NaN`/`inf`) samples in a channel, if present.
    pub fn nan_count(&self, id: ChannelId) -> Option<usize> {
        self.channels
            .get(&id)
            .map(|v| v.iter().filter(|x| !x.is_finite()).count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_round_trips() {
        assert_eq!(GridKind::parse("s"), Some(GridKind::S));
        assert_eq!(GridKind::parse(" t "), Some(GridKind::T));
        assert_eq!(GridKind::parse("x"), None);
        assert_eq!(GridKind::S.as_str(), "s");
        assert_eq!(GridKind::S.axis_channel(), ChannelId::S);
        assert_eq!(GridKind::T.axis_channel(), ChannelId::Time);
    }

    #[test]
    fn validity_and_counts() {
        let mut channels = BTreeMap::new();
        channels.insert(ChannelId::S, vec![0.0, 1.0, 2.0]);
        channels.insert(ChannelId::Speed, vec![10.0, f64::NAN, 30.0]);
        let t = Telemetry {
            grid: GridKind::S,
            channels,
            metadata: Vec::new(),
        };
        assert_eq!(t.len(), 3);
        assert!(!t.is_empty());
        assert_eq!(t.axis(), Some(&[0.0, 1.0, 2.0][..]));
        assert_eq!(
            t.validity_mask(ChannelId::Speed),
            Some(vec![true, false, true])
        );
        assert_eq!(t.nan_count(ChannelId::Speed), Some(1));
        assert_eq!(t.nan_count(ChannelId::S), Some(0));
        assert_eq!(t.nan_count(ChannelId::Gear), None);
    }
}
