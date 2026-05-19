// stanza-parseviewer.js
// Drop-in replacement for the original dagre-D3-based parse viewer.
// Uses a simple constituency-correct layout algorithm:
//   1. Leaves are assigned x-positions in sentence order (left to right).
//   2. Internal nodes are centered over their children's leaf span.
//   3. Y is determined by depth in the tree.
// This ensures words always read left-to-right, matching the sentence order.
//
// API is identical to the original: stanza-brat.js needs no changes.

//'use strict';

var ParseViewer = function(params) {
  this.selector = params.selector;
  this.container = $(this.selector);
  this.fitToGraph = true;
  this.onClickNodeCallback = params.onClickNodeCallback;
  this.onHoverNodeCallback = params.onHoverNodeCallback;
  this.init();
  return this;
};

ParseViewer.MIN_WIDTH = 100;
ParseViewer.MIN_HEIGHT = 100;
ParseViewer.prototype.constructor = ParseViewer;

ParseViewer.prototype.getAutoWidth = function () {
  return Math.max(ParseViewer.MIN_WIDTH, this.container.width());
};
ParseViewer.prototype.getAutoHeight = function () {
  return Math.max(ParseViewer.MIN_HEIGHT, this.container.height() - 20);
};

ParseViewer.prototype.init = function () {
  var canvasWidth = this.getAutoWidth();
  var canvasHeight = this.getAutoHeight();
  this.parseElem = d3.select(this.selector)
    .append('svg')
    .attr({'width': canvasWidth, 'height': canvasHeight})
    .style({'width': canvasWidth, 'height': canvasHeight});
  this.graph = null;
  this.graphRendered = false;
  this.controls = $('<div class="text"></div>');
  this.container.append(this.controls);
};

// ── Layout constants ───────────────────────────────────────────────
var NODE_WIDTH  = 60;   // px reserved per leaf column (tune if needed)
var NODE_HEIGHT = 48;   // px per tree level
var PADDING_X   = 30;
var PADDING_TOP = 20;
var BOX_W       = 50;
var BOX_H       = 24;

// ── Tree layout helpers ────────────────────────────────────────────

// Assign .depth to every node; terminal nodes also get .wordDepth (one below).
function assignDepths(node, depth) {
  node.depth = depth;
  if (node.isTerminal) {
    node.wordDepth = depth + 1;
  } else if (node.children) {
    for (var i = 0; i < node.children.length; i++) {
      assignDepths(node.children[i], depth + 1);
    }
  }
}

// Collect all terminal nodes in left-to-right order.
function collectLeaves(node, leaves) {
  if (node.isTerminal) {
    leaves.push(node);
  } else if (node.children) {
    for (var i = 0; i < node.children.length; i++) {
      collectLeaves(node.children[i], leaves);
    }
  }
}

// Return the leftmost leaf under a node.
function leftmostLeaf(node) {
  if (node.isTerminal) return node;
  return leftmostLeaf(node.children[0]);
}

// Return the rightmost leaf under a node.
function rightmostLeaf(node) {
  if (node.isTerminal) return node;
  return rightmostLeaf(node.children[node.children.length - 1]);
}

// Assign .x to internal nodes (bottom-up). Leaves must already have .x set.
function assignX(node) {
  if (node.isTerminal) return;  // x already set during leaf indexing
  if (node.children) {
    for (var i = 0; i < node.children.length; i++) {
      assignX(node.children[i]);
    }
    // Center this node over its leftmost and rightmost leaf descendants.
    var first = leftmostLeaf(node);
    var last  = rightmostLeaf(node);
    node.x = (first.x + last.x) / 2;
  }
}

// Return the maximum depth in the subtree (word nodes count as wordDepth).
function maxDepth(node) {
  if (node.isTerminal) return node.wordDepth;
  var d = node.depth;
  for (var i = 0; i < node.children.length; i++) {
    d = Math.max(d, maxDepth(node.children[i]));
  }
  return d;
}

function yForDepth(depth) {
  return depth * NODE_HEIGHT;
}

// Return the box width needed to fit a given text label, with a minimum of BOX_W.
function boxWidth(text) {
  return Math.max(BOX_W, text.length * 7 + 16);
}

// ── Drawing ────────────────────────────────────────────────────────

function drawEdges(svg, node) {
  var px = node.x;
  var py = yForDepth(node.depth);

  if (node.isTerminal) {
    var wy = yForDepth(node.wordDepth);
    svg.append('line')
      .attr({'x1': px, 'y1': py + BOX_H / 2 + 1,
             'x2': px, 'y2': wy - BOX_H / 2 - 1})
      .style({'stroke': '#999', 'stroke-width': 1.5});
  } else if (node.children) {
    for (var i = 0; i < node.children.length; i++) {
      var child = node.children[i];
      var cx = child.x;
      var cy = yForDepth(child.depth);
      svg.append('line')
        .attr({'x1': px, 'y1': py + BOX_H / 2 + 1,
               'x2': cx, 'y2': cy - BOX_H / 2 - 1})
        .style({'stroke': '#999', 'stroke-width': 1.5});
      drawEdges(svg, child);
    }
  }
}

function drawNodes(svg, node, scope) {
  var x = node.x;
  var y = yForDepth(node.depth);

  // Internal / POS node — green, matching the original brat style.
  var bw = boxWidth(node.label);
  var g = svg.append('g')
    .attr('transform', 'translate(' + x + ',' + y + ')')
    .attr('class', 'parse-RULE')
    .style('cursor', 'pointer');

  g.append('rect')
    .attr({'x': -bw / 2, 'y': -BOX_H / 2,
           'width': bw,  'height': BOX_H, 'rx': 5, 'ry': 5})
    .style({'fill': '#c6e5b3', 'stroke': '#5aaa3a', 'stroke-width': 1.5});

  g.append('text')
    .attr({'text-anchor': 'middle', 'dy': '0.35em'})
    .style({'font-size': '12px', 'font-family': 'monospace'})
    .text(node.label);

  if (scope.onClickNodeCallback) {
    g.on('click', (function(n) {
      return function() { scope.onClickNodeCallback(n); };
    })(node));
  }
  if (scope.onHoverNodeCallback) {
    g.on('mouseover', (function(n) {
      return function() { scope.onHoverNodeCallback(n); };
    })(node));
  }

  if (node.isTerminal) {
    // Word (leaf) node — yellow, matching the original style.
    var wy = yForDepth(node.wordDepth);
    var wbw = boxWidth(node.text);
    var wg = svg.append('g')
      .attr('transform', 'translate(' + x + ',' + wy + ')')
      .attr('class', 'parse-TERMINAL');

    wg.append('rect')
      .attr({'x': -wbw / 2, 'y': -BOX_H / 2,
             'width': wbw,  'height': BOX_H, 'rx': 5, 'ry': 5})
      .style({'fill': '#ffffc0', 'stroke': '#cccc60', 'stroke-width': 1.5});

    wg.append('text')
      .attr({'text-anchor': 'middle', 'dy': '0.35em'})
      .style({'font-size': '12px', 'font-family': 'monospace'})
      .text(node.text);
  } else if (node.children) {
    for (var i = 0; i < node.children.length; i++) {
      drawNodes(svg, node.children[i], scope);
    }
  }
}

// ── Public API (identical to original) ────────────────────────────

ParseViewer.prototype.showParses = function (roots) {
  var svg = this.parseElem;
  svg.selectAll('*').remove();
  if (roots.length === 0) return;

  // Save for onResize
  this.parse = roots;

  // 1. Assign depths to every node.
  for (var ri = 0; ri < roots.length; ri++) {
    assignDepths(roots[ri], 0);
  }

  // 2. Collect all leaves across all sentences, in order.
  var allLeaves = [];
  for (var ri = 0; ri < roots.length; ri++) {
    collectLeaves(roots[ri], allLeaves);
  }

  // 3. Assign x to leaves in sentence order, spacing by actual box width.
  for (var li = 0; li < allLeaves.length; li++) {
    allLeaves[li].x = li === 0
      ? boxWidth(allLeaves[li].text) / 2
      : allLeaves[li - 1].x + Math.max(NODE_WIDTH,
          boxWidth(allLeaves[li - 1].text) / 2 + boxWidth(allLeaves[li].text) / 2 + 10);
  }

  // 4. Assign x to internal nodes (bottom-up).
  for (var ri = 0; ri < roots.length; ri++) {
    assignX(roots[ri]);
  }

  // 5. Compute canvas size.
  var totalDepth = 0;
  for (var ri = 0; ri < roots.length; ri++) {
    totalDepth = Math.max(totalDepth, maxDepth(roots[ri]));
  }

  var totalWidth = allLeaves.length > 0
    ? allLeaves[allLeaves.length - 1].x + boxWidth(allLeaves[allLeaves.length - 1].text) / 2
    : BOX_W;
  var totalHeight = yForDepth(totalDepth) + BOX_H + PADDING_TOP * 2;

  svg.attr({'width':  totalWidth  + PADDING_X   * 2,
            'height': totalHeight});
  svg.style({'width': totalWidth  + PADDING_X   * 2,
             'height': totalHeight});

  // 6. Render into a translated group.
  var svgGroup = svg.append('g')
    .attr('transform', 'translate(' + PADDING_X + ',' + PADDING_TOP + ')');

  // Draw edges beneath nodes.
  for (var ri = 0; ri < roots.length; ri++) {
    drawEdges(svgGroup, roots[ri]);
  }
  // Draw nodes on top.
  for (var ri = 0; ri < roots.length; ri++) {
    drawNodes(svgGroup, roots[ri], this);
  }

  this.graphRendered = true;
};

ParseViewer.prototype.showParse = function (root) {
  this.showParses([root]);
};

ParseViewer.prototype.showAnnotation = function (annotation) {
  var parses = [];
  for (var i = 0; i < annotation.sentences.length; i++) {
    var s = annotation.sentences[i];
    if (s && s.parseTree) {
      parses.push(s.parseTree);
    }
  }
  this.showParses(parses);
};

ParseViewer.prototype.onResize = function () {
  if (this.parse && this.parse.length > 0) {
    this.showParses(this.parse);
  }
};
