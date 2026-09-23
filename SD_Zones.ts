# Supply & Demand Zones (James Sibbet Demand Index) -- ThinkScript
# Ported from the Pine v6 "Supply & Demand Zones" indicator (itself a port of the
# original TOS study). Core Demand Index / zone-stamping / resistance-band math is
# byte-for-byte equivalent to the Pine version. Two things ThinkScript genuinely
# cannot reproduce, called out where relevant:
#   1. Zone HISTORY (Pine's "Zones kept per side" > 1). ThinkScript has no general
#      object-array type, so only the CURRENT zone per side can be tracked/drawn --
#      which is also the Pine script's own default (maxZones = 1), so this is not a
#      loss versus how you'd normally run it.
#   2. True rectangle/box drawing. TOS has no box.new() equivalent, so zones are
#      drawn with AddCloud (a continuous fill between two plots) instead of a boxed
#      rectangle -- visually a filled band rather than a bordered box, but it
#      occupies the same price/bar region.
#
# Bonus: TOS's HighestAll() is genuinely non-causal (it scans the WHOLE loaded
# chart, not just bars up to "now"), which is exactly the behavior the Pine port
# had to work around with a last-bar rescan loop. Here we can just call it directly.

declare upper;

input lookback = 5;          # smoothing period for pressure averages / volume ratio
input sensitivity = 0.35;    # Demand Index threshold; lower = more zones, higher = fewer/stronger
input atrMult = 1.5;         # resistance band ATR multiplier
input showDemandZone = yes;
input showSupplyZone = yes;
input onBreak = {default Ignore, Remove, "Fade and keep"};   # what happens when price closes through a zone
input showResistanceBand = yes;
input resistanceBandFallback = no;   # draw the band from live ATR before the first breakout (TOS normally hides it until then)
input showLabels = yes;

def breakRemove = onBreak == onBreak."Remove";
def breakFade = onBreak == onBreak."Fade and keep";

# ------------------------------------------------------------------------------
#  DEMAND INDEX ENGINE
# ------------------------------------------------------------------------------
def wc = (high + low + 2 * close) * 0.25;
def wcRate = (wc - wc[1]) / Min(wc, wc[1]);

def rngAvg = Average(Highest(high, 2) - Lowest(low, 2), lookback);
def volatilityRatio = if rngAvg != 0 then 3 * wc / rngAvg * AbsValue(wcRate) else 0.0;

def volAvg = Average(volume, lookback);
def volumeRatio = if volAvg != 0 then volume / volAvg else 0.0;
def volPerRange = volumeRatio / Exp(Min(88, volatilityRatio));

def atrDI = Average(TrueRange(high, close, low), lookback) * atrMult;

def buyP = if wcRate > 0 then volumeRatio else volPerRange;
def sellP = if wcRate > 0 then volPerRange else volumeRatio;

rec buyPres = if IsNaN(buyPres[1]) then 0.0 else (buyPres[1] * (lookback - 1) + buyP) / lookback;
rec sellPres = if IsNaN(sellPres[1]) then 0.0 else (sellPres[1] * (lookback - 1) + sellP) / lookback;

def DI = if (sellPres - buyPres) > 0
    then -(if sellPres != 0 then buyPres / sellPres else 1.0)
    else (if buyPres != 0 then sellPres / buyPres else 1.0);
def DMI = if DI < 0 then -1 - DI else 1 - DI;

# ------------------------------------------------------------------------------
#  ZONE TRACKING  (cross to stamp, ratchet while extreme -- unchanged TOS logic)
# ------------------------------------------------------------------------------
def dTrig = DMI crosses below -sensitivity;
rec demandL = if dTrig then low
    else if DMI < -0.2 and !IsNaN(demandL[1]) and low < demandL[1] then low
    else demandL[1];
def dSet = !IsNaN(demandL) and low == demandL;
rec demandH = if dSet then high else demandH[1];
rec demandX = if dSet then BarNumber() else demandX[1];

def sTrig = DMI crosses above sensitivity;
rec supplyH = if sTrig then high
    else if DMI > 0.5 and !IsNaN(supplyH[1]) and high > supplyH[1] then high
    else supplyH[1];
def sSet = !IsNaN(supplyH) and high == supplyH;
rec supplyL = if sSet then low else supplyL[1];
rec supplyX = if sSet then BarNumber() else supplyX[1];

# Zone "alive" state: a demand zone dies on a close below it, a supply zone on a
# close above it. Resets to alive when a fresh zone stamps.
rec demandDead = if dTrig then no else if breakRemove or breakFade then (demandDead[1] or close < demandL) else no;
rec supplyDead = if sTrig then no else if breakRemove or breakFade then (supplyDead[1] or close > supplyH) else no;

def demandLive = showDemandZone and !IsNaN(demandL) and !demandDead;
def supplyLive = showSupplyZone and !IsNaN(supplyH) and !supplyDead;
def demandShown = showDemandZone and !IsNaN(demandL) and (!demandDead or breakFade);
def supplyShown = showSupplyZone and !IsNaN(supplyH) and (!supplyDead or breakFade);

# ------------------------------------------------------------------------------
#  RESISTANCE BAND  (TOS's HighestAll() is genuinely non-causal, so this is the
#  literal "TOS match" behavior the Pine port had to rescan history to reproduce)
# ------------------------------------------------------------------------------
def SH = HighestAll(supplyH);
rec res = if close crosses above SH then atrDI else res[1];
def resUse = if IsNaN(res) and resistanceBandFallback then atrDI else res;
def bandOn = showResistanceBand and !IsNaN(supplyX) and !IsNaN(supplyH) and !IsNaN(resUse);

# ------------------------------------------------------------------------------
#  PLOTS  (AddCloud fills stand in for Pine's box.new rectangles)
# ------------------------------------------------------------------------------
# Edge lines are left VISIBLE (not hidden) so the AddCloud fill reads with a
# border, closer to the bordered-box look TradingView draws with box.new().
plot DemandTop = if demandShown and !demandDead then demandH else Double.NaN;
plot DemandBot = if demandShown and !demandDead then demandL else Double.NaN;
DemandTop.SetDefaultColor(CreateColor(0, 188, 212));
DemandBot.SetDefaultColor(CreateColor(0, 188, 212));
DemandTop.SetLineWeight(1);
DemandBot.SetLineWeight(1);
AddCloud(DemandTop, DemandBot, CreateColor(0, 188, 212), CreateColor(0, 188, 212));

plot DemandTopFaded = if demandShown and demandDead then demandH else Double.NaN;
plot DemandBotFaded = if demandShown and demandDead then demandL else Double.NaN;
DemandTopFaded.SetDefaultColor(CreateColor(0, 90, 105));
DemandBotFaded.SetDefaultColor(CreateColor(0, 90, 105));
DemandTopFaded.SetLineWeight(1);
DemandBotFaded.SetLineWeight(1);
AddCloud(DemandTopFaded, DemandBotFaded, CreateColor(0, 90, 105), CreateColor(0, 90, 105));

plot SupplyTop = if supplyShown and !supplyDead then supplyH else Double.NaN;
plot SupplyBot = if supplyShown and !supplyDead then supplyL else Double.NaN;
SupplyTop.SetDefaultColor(CreateColor(255, 193, 7));
SupplyBot.SetDefaultColor(CreateColor(255, 193, 7));
SupplyTop.SetLineWeight(1);
SupplyBot.SetLineWeight(1);
AddCloud(SupplyTop, SupplyBot, CreateColor(255, 193, 7), CreateColor(255, 193, 7));

plot SupplyTopFaded = if supplyShown and supplyDead then supplyH else Double.NaN;
plot SupplyBotFaded = if supplyShown and supplyDead then supplyL else Double.NaN;
SupplyTopFaded.SetDefaultColor(CreateColor(130, 100, 5));
SupplyBotFaded.SetDefaultColor(CreateColor(130, 100, 5));
SupplyTopFaded.SetLineWeight(1);
SupplyBotFaded.SetLineWeight(1);
AddCloud(SupplyTopFaded, SupplyBotFaded, CreateColor(130, 100, 5), CreateColor(130, 100, 5));

plot BandTop = if bandOn then supplyH + 2 * resUse else Double.NaN;
plot BandBot = if bandOn then supplyH + resUse else Double.NaN;
BandTop.SetDefaultColor(CreateColor(239, 83, 80));
BandBot.SetDefaultColor(CreateColor(239, 83, 80));
BandTop.SetLineWeight(1);
BandBot.SetLineWeight(1);
BandTop.SetStyle(Curve.SHORT_DASH);
BandBot.SetStyle(Curve.SHORT_DASH);
AddCloud(BandTop, BandBot, CreateColor(239, 83, 80), CreateColor(239, 83, 80));

# ------------------------------------------------------------------------------
#  LABELS  (AddChartBubble is TOS's real equivalent of Pine's label.new -- a
#  text tag anchored to a specific bar/price, unlike AddLabel's fixed corner
#  strip. Shown only on the most recent bar, at the zone's near edge, matching
#  where TradingView draws its "Demand 29513.00" / "Supply ..." tag.)
# ------------------------------------------------------------------------------
def isLastBar = !IsNaN(close) and IsNaN(close[-1]);

AddChartBubble(showLabels and isLastBar and demandLive, demandH,
    "Demand " + AsText(Round(demandH, 2)), CreateColor(0, 188, 212), no);
AddChartBubble(showLabels and isLastBar and supplyLive, supplyL,
    "Supply " + AsText(Round(supplyL, 2)), CreateColor(255, 193, 7), yes);

# ------------------------------------------------------------------------------
#  ALERT-READY CONDITIONS  (build TOS alerts off these via "Study Alert" / right-
#  click a plot > Create Alert; ThinkScript doesn't have Pine's alertcondition())
# ------------------------------------------------------------------------------
def inD = demandLive and close <= demandH and close >= demandL;
def inS = supplyLive and close <= supplyH and close >= supplyL;
plot IntoDemand = inD and !inD[1];
plot IntoSupply = inS and !inS[1];
plot DemandBroken = demandLive[1] and !demandLive;
plot SupplyBroken = supplyLive[1] and !supplyLive;
plot AboveResistanceBand = bandOn and close crosses above supplyH + 2 * resUse;
IntoDemand.Hide();
IntoSupply.Hide();
DemandBroken.Hide();
SupplyBroken.Hide();
AboveResistanceBand.Hide();

# Data-window reference values
plot DemandIndex = DMI;
DemandIndex.Hide();
