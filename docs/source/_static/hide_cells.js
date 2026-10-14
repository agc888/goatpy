// Collapse selected notebook code cells into a click-to-expand block.
// A cell is collapsed if its source starts with one of the names below
// (ignoring leading quotes/whitespace, so a """docstring""" header matches).
document.addEventListener("DOMContentLoaded", function () {
  var markers = [
    "plot_mean_spectra.py",
    "glycan_calibration_benchmark.py",
  ];

  document.querySelectorAll("div.nbinput").forEach(function (cell) {
    var area = cell.querySelector(".input_area");
    if (!area) return;

    var text = area.textContent.replace(/^["'\s]+/, "");
    var marker = markers.find(function (m) { return text.indexOf(m) === 0; });
    if (!marker) return;

    var details = document.createElement("details");
    details.className = "collapsed-cell";
    var summary = document.createElement("summary");
    summary.textContent = "Show code (" + marker + ")";
    summary.style.cursor = "pointer";
    details.appendChild(summary);
    cell.parentNode.insertBefore(details, cell);

    // move the input cell (and any output cells directly after it) inside
    var node = cell;
    while (node) {
      var next = node.nextElementSibling;
      details.appendChild(node);
      if (!next || !next.classList.contains("nboutput")) break;
      node = next;
    }
  });
});