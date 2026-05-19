<?php
// Plot directory browser — improved fork of gpetrucc's classic index.php.
// Drop into any directory of PNG/PDF plots and it renders a searchable gallery
// with subdirectory navigation, breadcrumbs, dark-mode support, and a
// recursive (subtree) view toggle.

function h($s) { return htmlspecialchars((string)$s, ENT_QUOTES, 'UTF-8'); }
function get($k, $d = '') { return isset($_GET[$k]) ? $_GET[$k] : $d; }

$cwd        = getcwd();
$match      = (string) get('match', '');
$is_regexp  = !empty($_GET['regexp']);
$noplots    = !empty($_GET['noplots']);
$recursive  = !empty($_GET['recursive']);
$sort_mode  = get('sort', 'name'); // name | mtime

function name_match($filename, $match, $is_regexp) {
    if ($match === '') return true;
    if ($is_regexp) {
        return @preg_match('/' . str_replace('/', '\\/', $match) . '/i', $filename) === 1;
    }
    return fnmatch('*' . $match . '*', $filename, FNM_CASEFOLD);
}

function gather_pngs($recursive) {
    if (!$recursive) {
        $r = glob('*.png');
        return $r ?: [];
    }
    $out = [];
    $stack = ['.'];
    while ($stack) {
        $d = array_pop($stack);
        foreach (glob(rtrim($d, '/') . '/*') as $p) {
            $base = basename($p);
            if ($base[0] === '.') continue;
            if (is_dir($p)) {
                if (!preg_match('/private/i', $base)) $stack[] = $p;
            } elseif (str_ends_with(strtolower($p), '.png')) {
                $out[] = ltrim(preg_replace('#^\./#', '', $p), '/');
            }
        }
    }
    return $out;
}

// Breadcrumbs relative to /eos/user/.../www
$path_parts  = explode('/', trim($cwd, '/'));
$crumbs_html = '';
$accum = '';
$depth_to_root = 0;
foreach ($path_parts as $part) {
    $accum .= '/' . $part;
    if ($part === 'www') {
        $crumbs_html .= '<span class="crumb-sep">›</span><a href="/' . h(ltrim($accum, '/')) . '/">' . h($part) . '</a>';
        $depth_to_root = 0;
        continue;
    }
    if ($crumbs_html !== '') {
        $rel = str_repeat('../', max($depth_to_root, 0));
        $crumbs_html .= '<span class="crumb-sep">›</span><a href="' . h($rel) . '">' . h($part) . '</a>';
    }
    $depth_to_root++;
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title><?= h(basename($cwd)) ?> — plots</title>
<style>
:root {
    --bg: #fafafa; --fg: #222; --muted: #666;
    --card-bg: #fff; --card-br: #ddd;
    --accent: #5b21b6; --accent-hov: #ef4444;
    --code-bg: #f1f1f1;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #1a1a1a; --fg: #e6e6e6; --muted: #999;
        --card-bg: #242424; --card-br: #333;
        --accent: #a78bfa; --accent-hov: #f87171;
        --code-bg: #2a2a2a;
    }
    img { background: #ddd; }
}
* { box-sizing: border-box; }
body {
    margin: 0; padding: 1.2em 1.6em;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Candara, sans-serif;
    font-size: 10pt; line-height: 1.45;
    background: var(--bg); color: var(--fg);
}
header { border-bottom: 1px solid var(--card-br); padding-bottom: 0.6em; margin-bottom: 0.8em; }
.crumbs { font-size: 11pt; color: var(--muted); }
.crumb-sep { margin: 0 0.4em; color: var(--muted); }
a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-hov); text-decoration: underline; }
h1 { font-size: 14pt; margin: 0.2em 0 0.4em 0; font-weight: 600; word-break: break-all; }
h2 { font-size: 12pt; margin: 1.2em 0 0.5em 0; color: var(--muted); font-weight: 600;
     text-transform: uppercase; letter-spacing: 0.04em; }
.toolbar {
    display: flex; flex-wrap: wrap; gap: 0.5em 1em; align-items: center;
    padding: 0.5em 0.7em; background: var(--card-bg);
    border: 1px solid var(--card-br); border-radius: 6px;
    margin-bottom: 1em;
}
.toolbar input[type=text] {
    padding: 0.3em 0.5em; font: inherit;
    background: var(--bg); color: var(--fg);
    border: 1px solid var(--card-br); border-radius: 4px;
}
.toolbar button, .toolbar select {
    padding: 0.3em 0.7em; font: inherit;
    background: var(--bg); color: var(--fg);
    border: 1px solid var(--card-br); border-radius: 4px; cursor: pointer;
}
.toolbar label { user-select: none; }
.dirgrid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 0.4em; margin-bottom: 0.6em;
}
.dirgrid a {
    display: block; padding: 0.5em 0.7em;
    background: var(--card-bg); border: 1px solid var(--card-br); border-radius: 4px;
    font-weight: 500;
}
.dirgrid a:hover { border-color: var(--accent); }
.picgrid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 0.8em;
}
.pic {
    background: var(--card-bg);
    border: 1px solid var(--card-br); border-radius: 6px;
    padding: 0.4em 0.5em 0.5em 0.5em;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.pic h3 { font-size: 10pt; margin: 0.2em 0 0.4em 0; word-break: break-all; font-weight: 500; }
.pic img { width: 100%; height: auto; display: block; border: 1px solid var(--card-br); border-radius: 3px; }
.pic .meta { font-size: 8.5pt; color: var(--muted); margin-top: 0.4em; display: flex; flex-wrap: wrap; gap: 0.4em 0.7em; }
.pic .meta a { font-weight: 500; }
.readme { background: var(--code-bg); padding: 0.8em; border-radius: 4px; font-size: 10pt; }
ul.files { list-style: none; padding: 0; margin: 0; }
ul.files li { padding: 0.15em 0; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9.5pt; }
.empty { color: var(--muted); font-style: italic; }
.count { color: var(--muted); font-weight: normal; font-size: 10pt; }
</style>
</head>
<body>
<header>
    <div class="crumbs"><?= $crumbs_html ?></div>
    <h1><?= h($cwd) ?></h1>
</header>

<form class="toolbar" method="get">
    <label>Filter:
        <input type="text" name="match" size="24" value="<?= h($match) ?>" placeholder="substring or regex">
    </label>
    <label><input type="checkbox" name="regexp" value="1" <?= $is_regexp ? 'checked' : '' ?>> regex</label>
    <label><input type="checkbox" name="recursive" value="1" <?= $recursive ? 'checked' : '' ?>> recursive</label>
    <label><input type="checkbox" name="noplots" value="1" <?= $noplots ? 'checked' : '' ?>> hide plots</label>
    <label>Sort:
        <select name="sort">
            <option value="name"  <?= $sort_mode === 'name'  ? 'selected' : '' ?>>name</option>
            <option value="mtime" <?= $sort_mode === 'mtime' ? 'selected' : '' ?>>newest</option>
        </select>
    </label>
    <button type="submit">Apply</button>
    <?php if ($match !== '' || $is_regexp || $recursive || $noplots || $sort_mode !== 'name'): ?>
        <a href="?">reset</a>
    <?php endif; ?>
</form>

<?php
$dirs = [];
foreach (glob('*') as $f) {
    if (is_dir($f) && $f[0] !== '.' && !preg_match('/private/i', $f)) $dirs[] = $f;
}
sort($dirs);
if ($dirs):
?>
<h2>Directories <span class="count">(<?= count($dirs) ?>)</span></h2>
<div class="dirgrid">
<?php foreach ($dirs as $d): ?>
    <a href="<?= h(rawurlencode($d)) ?>/">📁 <?= h($d) ?></a>
<?php endforeach; ?>
</div>
<?php endif; ?>

<?php
foreach (['00_README.txt', 'README.txt', 'readme.txt', 'README.md'] as $rd) {
    if (file_exists($rd)) {
        echo "<h2>" . h($rd) . "</h2><pre class='readme'>";
        echo h(file_get_contents($rd));
        echo "</pre>";
        break;
    }
}
?>

<h2>Plots</h2>
<div>
<?php
$displayed = [];
if ($noplots) {
    echo "<p class='empty'>Plots hidden.</p>";
} else {
    $other_exts = ['.pdf', '.eps', '.root', '.txt', '.cxx', '.info'];
    $filenames  = gather_pngs($recursive);
    $filenames  = array_values(array_filter($filenames, fn($f) => name_match($f, $match, $is_regexp)));

    if ($sort_mode === 'mtime') {
        usort($filenames, fn($a, $b) => @filemtime($b) <=> @filemtime($a));
    } else {
        sort($filenames);
    }

    if (!$filenames) {
        echo "<p class='empty'>No matching plots.</p>";
    } else {
        echo "<p class='count'>" . count($filenames) . " plot" . (count($filenames) === 1 ? '' : 's') . "</p>";
        echo "<div class='picgrid'>";
        foreach ($filenames as $filename) {
            $displayed[] = $filename;
            $shown = str_replace('_', '_<wbr>', h($filename));
            $href  = h(implode('/', array_map('rawurlencode', explode('/', $filename))));
            $mtime = @filemtime($filename);
            $mtxt  = $mtime ? date('Y-m-d H:i', $mtime) : '';

            echo "<div class='pic'>";
            echo "<h3><a href=\"$href\">$shown</a></h3>";
            echo "<a href=\"$href\"><img loading=\"lazy\" src=\"$href\"></a>";
            $others = [];
            foreach ($other_exts as $ex) {
                $alt = preg_replace('/\.png$/i', $ex, $filename);
                if ($alt !== $filename && file_exists($alt)) {
                    $alt_href = h(implode('/', array_map('rawurlencode', explode('/', $alt))));
                    $others[] = "<a href=\"$alt_href\">[" . h(ltrim($ex, '.')) . "]</a>";
                    if ($ex !== '.txt') $displayed[] = $alt;
                }
            }
            echo "<div class='meta'>";
            if ($mtxt) echo "<span>$mtxt</span>";
            if ($others) echo implode('', $others);
            echo "</div>";
            echo "</div>";
        }
        echo "</div>";
    }
}
?>
</div>

<?php
$leftovers = [];
foreach (glob('*') as $f) {
    if (is_dir($f)) continue;
    if (in_array($f, $displayed, true)) continue;
    if (!name_match($f, $match, $is_regexp)) continue;
    if ($f === 'index.php') continue;
    $leftovers[] = $f;
}
sort($leftovers);
if ($leftovers):
?>
<h2>Other files <span class="count">(<?= count($leftovers) ?>)</span></h2>
<ul class="files">
<?php foreach ($leftovers as $f): ?>
    <li><a href="<?= h(rawurlencode($f)) ?>"><?= h($f) ?></a></li>
<?php endforeach; ?>
</ul>
<?php endif; ?>

</body>
</html>
