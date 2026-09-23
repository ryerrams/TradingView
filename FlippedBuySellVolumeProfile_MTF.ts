# Flipped Buy/Sell Pressure Volume Profile
#
# Uses the Volume Profile embedded in MonkeyBars so the histogram faces
# left from the right expansion area. The MonkeyBars/TPO display is hidden.
#
# Buy/sell pressure is an estimate based on each bar's closing location
# within its high-low range. It is not true bid/ask aggressor volume.
#
# Also adds a multi-timeframe Buy% strip (1m/5m/15m/30m/1h), bucketed
# from the chart's own bars via wall-clock time. Accurate multi-TF
# readings require running this study on a 1-minute chart.

declare upper;

input pricePerRowHeightMode = {
    default AUTOMATIC,
    TICKSIZE,
    CUSTOM
};
input customRowHeight = 1.0;

input timePerProfile = {
    default CHART,
    MINUTE,
    HOUR,
    DAY,
    WEEK,
    MONTH,
    "OPT EXP",
    BAR
};

input multiplier = 1;
input onExpansion = yes;
input profiles = 1000;
input showPointOfControl = yes;
input showValueArea = yes;
input showValueAreaLines = yes;
input valueAreaPercent = 70;
input opacity = 50;

input pressureThreshold = 52.0;
input showPressureLabel = yes;
input showMultiTFStrip = yes;

#-----------------------------
# Profile period calculation
#-----------------------------

def period;
def yyyymmdd = GetYYYYMMDD();
def seconds = SecondsFromTime(0);
def month = GetYear() * 12 + GetMonth();
def dayNumber =
    DaysFromDate(First(yyyymmdd)) +
    GetDayOfWeek(First(yyyymmdd));
def dom = GetDayOfMonth(yyyymmdd);
def dow = GetDayOfWeek(yyyymmdd - dom + 1);
def expirationDay = (if dow > 5 then 27 else 20) - dow;
def optionExpirationPeriod = month + (dom > expirationDay);

switch (timePerProfile) {
case CHART:
    period = 0;
case MINUTE:
    period = Floor(seconds / 60 + dayNumber * 24 * 60);
case HOUR:
    period = Floor(seconds / 3600 + dayNumber * 24);
case DAY:
    period = CountTradingDays(
        Min(First(yyyymmdd), yyyymmdd),
        yyyymmdd
    ) - 1;
case WEEK:
    period = Floor(dayNumber / 7);
case MONTH:
    period = Floor(month - First(month));
case "OPT EXP":
    period = optionExpirationPeriod - First(optionExpirationPeriod);
case BAR:
    period = BarNumber() - 1;
}

#-----------------------------
# Profile boundaries
#-----------------------------

def count = CompoundValue(
    1,
    if period != period[1]
    then (count[1] + period - period[1]) % multiplier
    else count[1],
    0
);

def startNewProfile = CompoundValue(
    1,
    count < count[1] + period - period[1],
    yes
);

#-----------------------------
# Profile row height
#-----------------------------

def rowHeight;

switch (pricePerRowHeightMode) {
case AUTOMATIC:
    rowHeight = PricePerRow.AUTOMATIC;
case TICKSIZE:
    rowHeight = PricePerRow.TICKSIZE;
case CUSTOM:
    rowHeight = customRowHeight;
}

#-----------------------------
# Estimated buy/sell pressure
#-----------------------------

def validBar = !IsNaN(close) and !IsNaN(volume);
def barRange = Max(high - low, TickSize());

# Closing near the high assigns more volume to estimated buying pressure;
# closing near the low assigns more to estimated selling pressure.
def estimatedBuyVolume =
    if validBar
    then if high > low
         then volume * (close - low) / barRange
         else volume * 0.5
    else 0;

def estimatedSellVolume =
    if validBar
    then volume - estimatedBuyVolume
    else 0;

def profileBuyVolume = CompoundValue(
    1,
    if startNewProfile
    then estimatedBuyVolume
    else if validBar
    then profileBuyVolume[1] + estimatedBuyVolume
    else profileBuyVolume[1],
    estimatedBuyVolume
);

def profileSellVolume = CompoundValue(
    1,
    if startNewProfile
    then estimatedSellVolume
    else if validBar
    then profileSellVolume[1] + estimatedSellVolume
    else profileSellVolume[1],
    estimatedSellVolume
);

def profileTotalVolume = profileBuyVolume + profileSellVolume;

def buyPercent =
    if profileTotalVolume > 0
    then 100 * profileBuyVolume / profileTotalVolume
    else 50;

def sellPercent = 100 - buyPercent;
def buyDominant = buyPercent >= pressureThreshold;
def sellDominant = sellPercent >= pressureThreshold;

#-----------------------------
# Multi-timeframe Buy% strip (1m/5m/15m/30m/1h)
#
# Each bucket accumulates estimatedBuyVolume/estimatedSellVolume from
# this chart's own bars using wall-clock time windows. A bucket smaller
# than or equal to the chart's own aggregation just collapses to the
# single-bar %; buckets above it need a 1-minute chart to be meaningful.
#-----------------------------

script BucketBuyPct {
    input bucketMinutes = 5;
    input nowSec = 0;
    input nowDay = 0;
    input buyVol = 0;
    input sellVol = 0;
    def bucketPeriod = Floor((nowSec + nowDay * 86400) / (bucketMinutes * 60));
    def newBucket = bucketPeriod != bucketPeriod[1];
    def bucketBV = CompoundValue(1, if newBucket then buyVol else bucketBV[1] + buyVol, buyVol);
    def bucketSV = CompoundValue(1, if newBucket then sellVol else bucketSV[1] + sellVol, sellVol);
    plot pct = 100 * bucketBV / (bucketBV + bucketSV);
}

def mtfPct1  = BucketBuyPct(1,  seconds, dayNumber, estimatedBuyVolume, estimatedSellVolume);
def mtfPct5  = BucketBuyPct(5,  seconds, dayNumber, estimatedBuyVolume, estimatedSellVolume);
def mtfPct15 = BucketBuyPct(15, seconds, dayNumber, estimatedBuyVolume, estimatedSellVolume);
def mtfPct30 = BucketBuyPct(30, seconds, dayNumber, estimatedBuyVolume, estimatedSellVolume);
def mtfPct60 = BucketBuyPct(60, seconds, dayNumber, estimatedBuyVolume, estimatedSellVolume);

def mtfSellThresh = 100 - pressureThreshold;
def chartAggMs = GetAggregationPeriod();

#-----------------------------
# Flipped Volume Profile
#-----------------------------

profile flippedProfile = MonkeyBars(
    # A constant interval prevents MonkeyBars from printing a long sequence
    # of multicolored TPO digits beside the embedded Volume Profile.
    timeInterval = 1,
    startNewProfile = startNewProfile,
    onExpansion = onExpansion,
    numberOfProfiles = profiles,
    pricePerRow = rowHeight,
    "the playground percent" = valueAreaPercent,
    "emphasize first digit" = no,
    volumeProfileShowStyle = MonkeyVolumeShowStyle.ALL,
    volumePercentVA = valueAreaPercent,
    "show initial balance" = no
);

# Calculate the actual Volume Profile value-area levels independently.
# This profile is not shown; it supplies VAH and VAL for the line plots.
profile volumeLevels = VolumeProfile(
    startNewProfile = startNewProfile,
    onExpansion = onExpansion,
    numberOfProfiles = profiles,
    pricePerRow = rowHeight,
    "value area percent" = valueAreaPercent
);

def expansionMode = CompoundValue(1, onExpansion, no);

def volumeVAH = CompoundValue(
    1,
    if IsNaN(volumeLevels.GetHighestValueArea()) and expansionMode
    then volumeVAH[1]
    else volumeLevels.GetHighestValueArea(),
    volumeLevels.GetHighestValueArea()
);

def volumeVAL = CompoundValue(
    1,
    if IsNaN(volumeLevels.GetLowestValueArea()) and expansionMode
    then volumeVAL[1]
    else volumeLevels.GetLowestValueArea(),
    volumeLevels.GetLowestValueArea()
);

def valueAreaPlotDomain = IsNaN(close) == onExpansion;

plot VAH =
    if showValueAreaLines and valueAreaPlotDomain
    then volumeVAH
    else Double.NaN;

plot VAL =
    if showValueAreaLines and valueAreaPlotDomain
    then volumeVAL
    else Double.NaN;

#-----------------------------
# Colors and display
#-----------------------------

DefineGlobalColor("Volume Profile", CreateColor(70, 130, 180));
DefineGlobalColor("Hidden Monkey Bars", Color.BLACK);
DefineGlobalColor("Buying Pressure", CreateColor(0, 175, 90));
DefineGlobalColor("Selling Pressure", CreateColor(210, 55, 55));
DefineGlobalColor("Balanced Pressure", CreateColor(120, 120, 120));
DefineGlobalColor("Value Area", CreateColor(160, 110, 210));
DefineGlobalColor("Point Of Control", Color.YELLOW);
DefineGlobalColor("VAH Line", Color.GREEN);
DefineGlobalColor("VAL Line", Color.RED);

VAH.SetPaintingStrategy(PaintingStrategy.HORIZONTAL);
VAH.SetDefaultColor(GlobalColor("VAH Line"));
VAH.SetLineWeight(2);
VAH.HideBubble();
VAH.HideTitle();

VAL.SetPaintingStrategy(PaintingStrategy.HORIZONTAL);
VAL.SetDefaultColor(GlobalColor("VAL Line"));
VAL.SetLineWeight(2);
VAL.HideBubble();
VAL.HideTitle();

AddLabel(
    showPressureLabel,
    "Estimated Buy " + Round(buyPercent, 1) +
    "% | Sell " + Round(sellPercent, 1) + "%",
    if buyDominant
    then GlobalColor("Buying Pressure")
    else if sellDominant
    then GlobalColor("Selling Pressure")
    else GlobalColor("Balanced Pressure")
);

AddLabel(showMultiTFStrip and chartAggMs != 60000,
    "⚠ Run on a 1-min chart for accurate multi-TF readings", Color.ORANGE);

AddLabel(showMultiTFStrip, "1m " + Round(mtfPct1, 0) + "%",
    if mtfPct1 >= pressureThreshold then GlobalColor("Buying Pressure")
    else if mtfPct1 <= mtfSellThresh then GlobalColor("Selling Pressure")
    else GlobalColor("Balanced Pressure"));
AddLabel(showMultiTFStrip, "5m " + Round(mtfPct5, 0) + "%",
    if mtfPct5 >= pressureThreshold then GlobalColor("Buying Pressure")
    else if mtfPct5 <= mtfSellThresh then GlobalColor("Selling Pressure")
    else GlobalColor("Balanced Pressure"));
AddLabel(showMultiTFStrip, "15m " + Round(mtfPct15, 0) + "%",
    if mtfPct15 >= pressureThreshold then GlobalColor("Buying Pressure")
    else if mtfPct15 <= mtfSellThresh then GlobalColor("Selling Pressure")
    else GlobalColor("Balanced Pressure"));
AddLabel(showMultiTFStrip, "30m " + Round(mtfPct30, 0) + "%",
    if mtfPct30 >= pressureThreshold then GlobalColor("Buying Pressure")
    else if mtfPct30 <= mtfSellThresh then GlobalColor("Selling Pressure")
    else GlobalColor("Balanced Pressure"));
AddLabel(showMultiTFStrip, "1h " + Round(mtfPct60, 0) + "%",
    if mtfPct60 >= pressureThreshold then GlobalColor("Buying Pressure")
    else if mtfPct60 <= mtfSellThresh then GlobalColor("Selling Pressure")
    else GlobalColor("Balanced Pressure"));

# The first color paints the remaining single-column MonkeyBars output the
# chart-background color. Change "Hidden Monkey Bars" under Global Colors if
# your expansion-area background is not black.
flippedProfile.Show(
    GlobalColor("Hidden Monkey Bars"),
    Color.CURRENT,
    Color.CURRENT,
    opacity,
    Color.CURRENT,
    Color.CURRENT,
    Color.CURRENT,
    GlobalColor("Volume Profile"),
    if showValueArea
    then GlobalColor("Value Area")
    else Color.CURRENT,
    if showPointOfControl
    then GlobalColor("Point Of Control")
    else Color.CURRENT
);
