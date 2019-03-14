var view_state = [0, 0, 1, 0, 0];
var tool_state = 0;
var current_qa = "";
var current_qa_data = {};
var current_att_data = {};
var selected_word = [];
var att_selected_info = [];
var expand_state = 0;
var keydown_state = 0;

var search_timeout = 0;
var adv_run_flag = false;

var predict_threshold = 0.05;
var tool_list = ["tool-embedding", "tool-attention", "tool-modeloutput", "tool-simquestion", "tool-adversarial"]

var filter = {};
var bookmark = [];

function init_filter_modal() {
	$("#modal-filter-label-true").attr('checked', filter["label"][0] == 1);
	$("#modal-filter-label-false").attr('checked', filter["label"][1] == 1);
	$("#modal-filter-pred-true").attr('checked', filter["pred"][0] == 1);
	$("#modal-filter-pred-false").attr('checked', filter["pred"][1] == 1);
	$("#modal-filter-prob-expr").val(filter["prob_expr"]);
	$("#modal-filter-shuffle").attr('checked', filter["shuffle"]);
	$("#modal-filter-limit-value").val(filter["limit"]);
}

$('#modal-filter-apply-btn').click(function() {
    filter["label"][0] = $("#modal-filter-label-true").is(':checked') ? 1 : 0;
    filter["label"][1] = $("#modal-filter-label-false").is(':checked') ? 1 : 0;
	filter["pred"][0] = $("#modal-filter-pred-true").is(':checked') ? 1 : 0;
    filter["pred"][1] = $("#modal-filter-pred-false").is(':checked') ? 1 : 0;
	filter["prob_expr"] = $("#modal-filter-prob-expr").val();
	filter["limit"] = parseInt($("#modal-filter-limit-value").val());
	filter["shuffle"] = $("#modal-filter-shuffle").is(':checked');
	localStorage.setItem("filter", JSON.stringify(filter));
	$('#modal-filter').modal('hide');
});


$(".nav-item:nth-child(1) .nav-link").addClass("active");
set_menu(0);

function set_menu(index) {
    $(".nav-item .nav-link").removeClass("active");
    $(".tool-content").css("display", "none");
    var selected = $(".nav-item:nth-child(" + (index + 1) + ") .nav-link");
    tool_state = index;
    var tool_id = "#" + tool_list[index];
    selected.addClass("active");
    $(tool_id).css("display", "block");
}

function load_sidebar(data) {
	$("#sidebar-id-list").empty();
		$("#sidebar-status-text").html("Total: " + data["data"].length + " (" + data["ratio"] + "%)");
		for (var i in data["data"]) {
			item = data["data"][i];
			ua_tag = item["uans"] == (item["pred"] > predict_threshold) ? "pos-ex" : "neg-ex";
			$("#sidebar-id-list").append(
				"<li id=\"qa-" + item["key"] + "\" class=\"qa-item " + ua_tag + "\">" +
				"<div>" + 
				"<div class=\"sidebar-qid text-monospace\">" + item["key"] + "</div>" +
				"<div class=\"sidebar-name\">" + item["name"] + "</div>" +
				"<div class=\"sidebar-question\">" + item["q"] + "</div>" +
				"</div></li>");
			$("#qa-" + item["key"]).click(function() {
				set_view($(this).attr("id").replace("qa-", ""), true);
			});
		}
}

function fetch_ids() {
    $("#sidebar-status-text").html("loading...");
	var query_text = {};
	query_text["filter"] = filter;
	var query = JSON.stringify(query_text);
	$.ajax("/ids", {
		data: query,
		contentType: 'application/json',
		type: 'POST',
		success: function(data) {
			load_sidebar(data);
		}
	});
}

function split_and_span(text) {
    text_words = text.split(" ");
    new_text_words = [];
	cnt = 0;
    for (var i in text_words) {
        new_text_words.push("<span_class=\"view-word\"idx=\""+cnt+"\"value=\"" + hash_value(text_words[i]) + "\">" + text_words[i] + "</span>");
		cnt += 1
    }
    return new_text_words;
}

String.prototype.hashCode = function() {
    var hash = 0,
        i, chr;
    if (this.length === 0) return hash;
    for (i = 0; i < this.length; i++) {
        chr = this.charCodeAt(i);
        hash = ((hash << 5) - hash) + chr;
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
};

function argsort(data, reverse) {
    var result = data
        .map((item, index) => [item, index])
        .sort(([a], [b]) => reverse ? b - a : a - b)
        .map(([, index]) => index);
    return result;
}


function hash_value(word) {
    return word.hashCode();
}



function set_em(result) {
    if (tool_state != 0) return;
    if (result.hasOwnProperty("error")) return;
    $("#tool-em-sim-list, #tool-em-vector-vis, #tool-em-space-vis").empty();

    var count = 1;
    words = result["word"];
    if (words.length > 1) count = 2;

    $("#tool-em-info-simlist").hide();
    $("#tool-em-info-simpair").hide();

    if (count == 1) {
        word_list = result["word_list"]
        dist_list = result["dist_list"]
        for (var i = 0; i < word_list.length; i++) {
            wid = "item-" + hash_value(word_list[i]);
            item = word_list[i] + " (" + dist_list[i].toFixed(3) + ")";
            $("#tool-em-sim-list").append("<li id=\"" + wid + "\" class=\"nav-item em-sim-word\">" + item + "</li>");
        }
        $("#tool-em-info-simlist").show();
    } else {
        $("#tool-em-simpair-result").text(result["sim"].toFixed(3))
        $("#tool-em-info-simpair").show();
    }


    // vector visualization
    var vector = result["embed"];
    var vector_size = vector[0].length;
    var margin = {
            top: 5,
            right: 0,
            bottom: 5,
            left: 100
        },
        width = 1600 - margin.left - margin.right,
        height = 35 * (count + 1) - margin.top - margin.bottom,
        itemHeight = 25,
        legendWidth = 250,
        buckets = 7,
        colors = ["#e51610", "#ffffff", "#1029e5"],
        label = words;

    var embed_svg = d3.select("#tool-em-vector-vis").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var wordLabels = embed_svg.selectAll(".wordLabel")
        .data(label)
        .enter().append("text")
        .text(function(d) {
            return d;
        })
        .attr("x", 0)
        .attr("y", function(d, i) {
            return (10 + itemHeight) * i;
        })
        .style("text-anchor", "end")
        .attr("transform", "translate(-6," + itemHeight / 1.5 + ")");

    var range = [-0.1, 0, 0.1];
    var colorScale = d3.scaleLinear()
        .domain(range)
        .range(colors);

    for (var i = 0; i < count; i++) {
        var y_margin = (10 + itemHeight) * i;
        vector_map = [];
        for (var j = 0; j < vector[i].length; j++) {
            vector_map.push({
                "x": j,
                "y": vector[i][j]
            });
        }

        var vector_area = embed_svg.append("g")
            .attr("class", "vector")
        vector_area.append("rect")
            .attr("y", y_margin)
            .attr("width", vector_size * 2 + 2)
            .attr("height", itemHeight)
            .attr("class", "back");

        var cards = vector_area.selectAll(".y" + i)
            .data(vector_map, function(d) {
                return d.x + "-" + d.y;
            });

        cards.enter().append("rect")
            .attr("x", function(d) {
                return 1 + (d.x) * 2;
            })
            .attr("y", function(d) {
                return 1 + y_margin;
            })
            .attr("class", "value bordered")
            .attr("width", 2)
            .attr("height", itemHeight - 2)
            .style("fill", function(d) {
                return colorScale(d.y);
            });

        cards.select("title").text(function(d) {
            return d.y;
        });
        cards.exit().remove();
    }

    var legend_area = embed_svg.append('defs')
        .append('svg:linearGradient')
        .attr('id', 'att-gradient')
        .attr('x1', '0%')
        .attr('y1', '100%')
        .attr('x2', '100%')
        .attr('y2', '100%')
        .attr('spreadMethod', 'pad');

    legend_area.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", "#e51610")
        .attr("stop-opacity", 1);

    legend_area.append("stop")
        .attr("offset", "50%")
        .attr("stop-color", "#ffffff")
        .attr("stop-opacity", 1);

    legend_area.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", "#1029e5")
        .attr("stop-opacity", 1);

    embed_svg.append("rect")
        .attr("width", legendWidth)
        .attr("height", 10)
        .attr("y", height - 25)
        .style("fill", "url(#att-gradient)")

    var y = d3.scaleLinear()
        .range([0, legendWidth / 2, legendWidth])
        .domain(range);

    var yAxis = d3.axisBottom()
        .scale(y)
        .ticks(5);

    embed_svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(0," + (height - 15) + ")")
        .call(yAxis)
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", height)
        .attr("dy", ".71em")
        .style("text-anchor", "end")
        .text("axis title");

    legend_area.exit().remove();

    // Embedding Space Visualization
    selected_word_text = [];
    for (var i = 0; i < selected_word.length; i++) {
        selected_word_text.push(selected_word[i].text())
    }
    var pca_data = result["pca_result"]["plot"];
    var pca_label = result["pca_result"]["label"];
    for (var i = 0; i < pca_data.length; i++) {
        if (selected_word_text.indexOf(pca_label[i]) != -1) {
            pca_data[i].push(1);
        } else {
            pca_data[i].push(0);
        }
        pca_data[i].push(pca_label[i]);
    }
    pca_data.reverse();

    var pca_width = 380;
    var pca_height = 380;

    var pca_svg = d3.select("#tool-em-space-vis")
        .append("svg")
        .attr("width", pca_width)
        .attr("height", pca_height);

    pca_width = 0.9 * pca_width;
    pca_height = 0.9 * pca_height;
    var margin = {
        top: 0,
        right: (0.05 * pca_width),
        bottom: (0.1 * pca_height),
        left: (0.05 * pca_height)
    };

    pca_svg.append("defs").append("clipPath")
        .attr("id", "clip")
        .append("rect")
        .attr("width", pca_width)
        .attr("height", pca_height);

    var max_x = d3.max(pca_data, function(d) {
        return d[0];
    })
    var min_x = d3.min(pca_data, function(d) {
        return d[0];
    })
    var max_y = d3.max(pca_data, function(d) {
        return d[1];
    })
    var min_y = d3.min(pca_data, function(d) {
        return d[1];
    })

    var x_range = Math.max(Math.abs(max_x), Math.abs(min_x));
    var y_range = Math.max(Math.abs(max_y), Math.abs(min_y));

    var xScale = d3.scaleLinear()
        .domain([-x_range * 1.5, x_range * 1.5])
        .range([0, pca_width]);
    var yScale = d3.scaleLinear()
        .domain([-y_range * 1.5, y_range * 1.5])
        .range([pca_width, 0]);

    var xAxis = d3.axisBottom(xScale)
        .tickFormat((domainn, number) => {
            return ""
        })
        .ticks(5, "s");
    var yAxis = d3.axisLeft(yScale)
        .tickFormat((domainn, number) => {
            return ""
        })
        .ticks(5, "s");

    var gX = pca_svg.append('g')
        .attr('class', 'pca-axis')
        .attr('transform', 'translate(' + margin.left + ',' + (margin.top + yScale(0)) + ')')
        .call(xAxis);
    var gY = pca_svg.append('g')
        .attr('class', 'pca-axis')
        .attr('transform', 'translate(' + (margin.left + xScale(0)) + ',' + margin.top + ')')
        .call(yAxis);

    // create axis objects

    var dataplot = pca_svg.selectAll("dp")
        .data(pca_data)
        .enter()
        .append('g')
		.attr("class", "emb-vis-point")
		.attr("title", function(d) {
			return d[3];
		});

    var points = dataplot
        .append("circle")
        .attr("cx", function(d) {
            return xScale(d[0]);
        })
        .attr("cy", function(d) {
            return yScale(d[1]);
        })
        .attr("id", function(d) {
            return hash_value(d[3]);
        })
        .attr("target", function(d, i) {
            return d[2] == 1 ? 1 : 0
        })
        .style("fill", function(d, i) {
            return d[2] == 1 ? "rgba(255,0,0,0.5)" : "rgba(0,0,255,0.5)"
        })
        .attr("r", 4);

    var texts = dataplot
        .append("text")
        .text(function(d) {
            return d[3];
        })
        .style("font-size", "12px")
        .attr("x", function(d, i) {
            return xScale(d[0]) + 0.1 * i;
        })
        .attr("y", function(d, i) {
            return yScale(d[1]) + 0.1 * i;
        })
        .attr("id", function(d) {
            return "label-" + hash_value(d[3]);
        });


    var zoom = d3.zoom()
        .scaleExtent([.5, 20])
        .extent([
            [0, 0],
            [pca_width, pca_height]
        ])
        .on("zoom", zoomed);
    pca_svg.append("rect")
        .attr("width", pca_width)
        .attr("height", pca_height)
        .style("fill", "none")
        .style("pointer-events", "all")
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
        .call(zoom);

    function zoomed() {
        var new_xScale = d3.event.transform.rescaleX(xScale);
        var new_yScale = d3.event.transform.rescaleY(yScale);

        gX.call(xAxis.scale(new_xScale));
        gY.call(yAxis.scale(new_yScale));
        points.data(pca_data)
            .attr('cx', function(d) {
                return new_xScale(d[0])
            })
            .attr('cy', function(d) {
                return new_yScale(d[1])
            });
        texts.data(pca_data)
            .attr('x', function(d) {
                return new_xScale(d[0])
            })
            .attr('y', function(d) {
                return new_yScale(d[1])
            });

    }

    $(".em-sim-word").hover(function() {
        var id = $(this).attr("id").replace("item-", "");
        var circle = $("#" + id);
        var text = $("#label-" + id)
        circle.attr("style", "fill: rgb(5, 214, 134);");
        text.attr("style", "font-size: 16px; fill: rgb(5, 214, 134);");
    }, function() {
        var id = $(this).attr("id").replace("item-", "");
        var circle = $("#" + id);
        var text = $("#label-" + id)
        var color = circle.attr("target") == 1 ? "rgba(255,0,0,0.5)" : "rgba(0,0,255,0.5)"
        circle.attr("style", "fill: " + color);
        text.attr("style", "font-size: 12px;");
    });

	$('[data-toggle="tooltip"]').tooltip();
    console.log(result)
}

function replaceAll(str, searchStr, replaceStr) {
    return str.split(searchStr).join(replaceStr);
}


function load_em_info() {
    display_word = [];
    if (selected_word.length > 0) {
		var query_text = {};
        for (var i = 0; i < selected_word.length; i++) {
            selected_word[i].addClass("selected-word");
            display_word.push(selected_word[i].text());
            query_text["word" + (i + 1)] = selected_word[i].text();
        }
        query_text["context"] = $("#view-context").text();
        query_text["conly"] = $("#tool-em-contextonly").is(":checked");
		query = JSON.stringify(query_text);
        $("#tool-em-title").html(display_word.join(" - "));
        $.ajax("/emb", {
            data: query,
            contentType: 'application/json',
            type: 'POST',
            success: function(data) {
                set_em(data);
                $("#tool-em-info").show();
            }
        });
    }
}

function set_view(qid, sidebar) {
    current_qa = qid;
    show_pp = view_state[0] > 0;
    show_oov = view_state[1] > 0;
    show_answer = view_state[2] > 0;
    show_att = view_state[3] > 0;
    selected_word = [];

    $.get("/qa?id=" + qid, function(result) {
		$("#view-init-video").remove();
		$("#view-init").hide();
        set_view_html(result, sidebar);
    });

	show_bookmark(qid);
	$("#view-head-bookmark").off("click");
	$("#view-head-bookmark").click(function(e) {
		change_bookmark(qid);
	});
}

function set_view_html(result, sidebar) {
	current_qa_data = result;
	curernt_qa_name = result["name"];
	pred_answer = result["pred"][0];
	pred_ua_prob = result["pred"][1];
	em_f1 = result["em_f1"];
	unanswerable_prob = pred_ua_prob > predict_threshold;

	head_class = unanswerable_prob == result["unanswerable"] ? "pos-ex" : "neg-ex";
	highlight_class = result["unanswerable"] == 0 ? "highlight-ans-p" : "highlight-ans-n";
	highlight_c = result["unanswerable"] == 0 ? "p" : "n";
	
	view_title_text = "[" + current_qa + "] " + curernt_qa_name + " | UA PRED " + pred_ua_prob.toFixed(3) + " LABEL " + result["unanswerable"];
	$("#view-head").removeClass();
	$("#view-head").addClass(head_class);

	$("#view-head-text").html(view_title_text);
	context_key = show_pp ? "context_p" : "context";
	question_key = show_pp ? "question_p" : "question";
	q_oov_key = show_pp ? "q_oov_p" : "q_oov";
	c_oov_key = show_pp ? "c_oov_p" : "c_oov";

	// view_context = split_and_span(result[context_key]).join(" ");
	back_context = result[context_key];
	question = split_and_span(result[question_key]).join(" ");
	answer_info = result["answers"];
	answer_text = [];
	answer_loc = [];
	if (show_pp) {
		answer_span = result["answer_p"];
		context_p = back_context.split(" ");
		a = []
		for (var i = answer_span[0]; i <= answer_span[1]; i++) {
			a.push(context_p[i]);
		}
		answer_text.push(a.join(" "));
	} else {
		var min = back_context.length;
		var max = 0;
		for (var i in answer_info) {
			ans = answer_info[i]["text"].trim();			
			ans_start = answer_info[i]["answer_start"];
			ans_end = ans_start + ans.length - 1;
			min_temp = min;
			max_temp = max;
			if (min > ans_start) min_temp = ans_start;
			if (max < ans_end) max_temp = ans_end;
			console.log("aaa " + (max_temp - min_temp + 1) + " "+ ans.length);
			if (max_temp - min_temp + 1 == ans.length) {
				min = min_temp;
				max = max_temp;
			}
			if (answer_text.indexOf(ans) < 0) {
				answer_text.push(ans);
			}
		}	
		answer_loc = [min, max];
	}

	// _" + highlight_c + "
	
	var htag_start = "[tag:strong]";
	var htag_end = "[/tag:strong]";
	var htag_start_real = "<strong class=\"highlight-ans-" + highlight_c + "\">";
	var htag_end_real = "</strong>";
	
	back_context_items = split_and_span(back_context);
	for (var i = 0; i < back_context_items.length; i++) {
		if (back_context_items[i].includes(htag_start)) {
			back_context_items[i] = htag_start_real + back_context_items[i].replace(htag_start, "")
			console.log(back_context_items[i]);
		} 
		if (back_context_items[i].includes(htag_end)) {
			back_context_items[i] = back_context_items[i].replace(htag_end, "") + htag_end_real;
			console.log(back_context_items[i]);
		}
	}
	
	back_context = back_context_items.join("<span> </span>");
	$("#view-context").html(replaceAll(back_context, "<span_class", "<span class"));
	$("#s_question").html(replaceAll(question, "<span_class", "<span class"));

	if (show_answer) {
		var context_span = $("#view-context span");
		if (show_pp) {
			answer_span = result["answer_p"];
			console.log("context_span/pp: " + context_span.length);
			console.log("answer_span: " + answer_span);
			var is_highlight = false;
			for (var i = 0; i < context_span.length; i++) {
				var word = context_span.eq(i);
				if (word.attr("idx") == answer_span[0]) {
					is_highlight = true;
				}
				if (is_highlight) {
					word.html(htag_start_real + word.html() + htag_end_real);
				}
				if (word.attr("idx") == answer_span[1]) {
					is_highlight = false;
					break;
				}
			}
		} else {
			console.log("context_span: " + context_span.length);
			console.log("answer_loc: " + JSON.stringify(answer_loc));
			var is_highlight = false;
			var start = 0;
			var end = 0;
			for (var i = 0; i < context_span.length; i++) {
				is_highlight = false;
				var word = context_span.eq(i);
				end = start + (word.text().length - 1);
				if (start >= answer_loc[0] && start <= answer_loc[1]) is_highlight = true;
				if (end >= answer_loc[0] && end <= answer_loc[1] ) is_highlight = true;
				
				if (is_highlight) {
					var word_text = word.text();
					var start_loc = Math.max(0, answer_loc[0] - start);
					var end_loc = Math.min(end - start, answer_loc[1] - start);
					word_text = word_text.substring(start_loc, end_loc + 1);
					word.html(word.html().replace(word_text, htag_start_real + word_text + htag_end_real));
				}
				start = end + 1;
			}
		}
	}
	
	if (show_oov) {
		var c_oov_info = result[c_oov_key];
		$("#view-context .view-word").each(function() {
			if (c_oov_info.includes(parseInt($(this).attr("idx")))) $(this).addClass("oov-word");
		});
		var q_oov_info = result[q_oov_key];
		$("#s_question .view-word").each(function() {
			if (q_oov_info.includes(parseInt($(this).attr("idx")))) $(this).addClass("oov-word");
		});
	} else {
		selected_word = [];
		$(".view-word").removeClass("oov-word")
	}

	if (unanswerable_prob) {
		$("#s_answer_pred").html("unanswerable (" + pred_answer + ")");
		$("#s_answer_pred").css("color", "#9f9f9f");
	} else {
		$("#s_answer_pred").html(pred_answer);
		$("#s_answer_pred").css("color", "#000000");
	}
	console.log(result["em_f1"]);
	var em_f1_html = "<span class=\"view-qa-metrics\">" 
			+ (result["em_f1"][0] == 1 ? "TRUE" : "FALSE") + " / " 
			+ (result["em_f1"][1]*100).toFixed(2) + "</span>"
	$("#s_em_f1").html(em_f1_html);
	if (result.hasOwnProperty("original_em_f1")) {
		console.log(result["original_em_f1"], result["em_f1"]);
	}
	
	if (result["unanswerable"]) {
		$("#s_answer_gold").html("unanswerable (" + answer_text.join(" / ") + ")");
		$("#s_answer_gold").css("color", "#9f9f9f");
	} else {
		$("#s_answer_gold").html(answer_text.join(" / "));
		$("#s_answer_gold").css("color", "#000000");
	}

	$(".view-word").click(function(e) {
		if (tool_state == 0) {
			query_text = {};
			$(".view-word").removeClass("selected-word");
			if (keydown_state != 18) {
				selected_word = [];
			} else if (keydown_state == 18 && selected_word.length > 1) {
				selected_word.shift();
			}
			if ($(this).hasClass("view-word")) selected_word.push($(this));
			load_em_info();
		}
		e.stopPropagation();
	});


	$('#tool-em-contextonly').change(function() {
		load_em_info();
	});

	$(".view-word").hover(function() {
		var value = $(this).attr("value");
		$(".view-word[value=" + value + "]").addClass("same_word");
	}, function() {
		$(".view-word").removeClass("same_word");
	});


	$(".view-word").dblclick(function(e) {
		if (tool_state == 4) {
			console.log("idx: " + $(".view-word").index(this));			
			var src = $(this).closest('div').attr('id');
			var index = $("#" + src + " .view-word").index(this);
			$("#modal-edit-target").text("[" + $(this).text() + "]");
			$("#modal-edit-new").val($(this).text());
			$("#modal-edit-idx").val(index);
			$("#modal-edit-src").val(src);
			$('#modal-edit').modal('show');
		}
		e.stopPropagation();
	});


	$("#view-context,#s_question").click(function(e) {
		$("#tool-em-sim-list, #tool-em-vector-vis, #tool-em-space-vis").empty();
		$(".view-word").removeClass("selected-word");
		selected_word = [];
		$("#tool-em-title").html("select word in context or question (alt key: compare two words)");
		$("#tool-em-info").hide();
		e.preventDefault();
	});

	if (sidebar && tool_state == 1) load_att_info();
	if (sidebar && tool_state == 2) load_output_info();
	if (sidebar && tool_state == 3) load_sent_info();
	if (sidebar && tool_state == 4 && adv_run_flag) adv_rule_apply();
}


function load_att_info() {
    $("#tool-att-weight").empty();
    query_text = {};
    $("#tool-att-loading").text("Drawing heatmap for Context2Question Attention...");
    $(".att-function").hide();
    query_text["c"] = $("#view-context").text();
    query_text["q"] = $("#s_question").text();
    if (query_text["c"].length > 0 && query_text["q"].length > 0) {
        query = JSON.stringify(query_text);
        $.ajax("/att", {
            data: query,
            contentType: 'application/json',
            type: 'POST',
            success: function(data) {
                current_att_data = data;
                draw_att_heatmap(data);
            }
        });
    }
}

function draw_att_heatmap(data) {
    $("#tool-att-weight").empty();
    $(".tooltip").remove();
    var flag = false;
    if (att_selected_info.length >= 2) {
        item1 = att_selected_info[0][0].__data__;
        item2 = att_selected_info[1][0].__data__;
        att_selected_info = [];

        var start_x = Math.min(item1.x, item2.x);
        var end_x = start_x + Math.abs(item1.x - item2.x);
        var start_y = Math.min(item1.y, item2.y);
        var end_y = start_y + Math.abs(item1.y - item2.y);

        flag = true;
    }

    data_list = [];
    for (var i = 0; i < data.length; i++) {
        for (var j = 0; j < data[i].length; j++) {
            if (!flag || (i >= start_x && i <= end_x && j >= start_y && j <= end_y))
                data_list.push({
                    "x": i,
                    "y": j,
                    "v": data[i][j]
                });
        }
    }

    var context_l = current_qa_data["context_p"].split(" ");
    var question_l = current_qa_data["question_p"].split(" ");

    if (flag) {
        context_l = context_l.slice(start_x, end_x + 1);
        question_l = question_l.slice(start_y, end_y + 1);
    }

    var margin = {
            top: 50,
            right: 0,
            bottom: 50,
            left: 30
        },
        size = 24,
        font_size = 16;
    width = Math.max(400, 150 + (size * context_l.length)),
        height = 60 + (size * (question_l.length)),
        legendWidth = 240;

    var svg = d3.select("#tool-att-weight").append("svg")
        .attr("id", "tool-att-weight-svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + (margin.left + 80) + "," + (margin.top + 50) + ")");

    var contextLabels = svg.append("g").selectAll(".context-label")
        .data(context_l)
        .enter().append("text")
        .text(function(d) {
            return d;
        })
        .attr("x", function(d, i) {
            return size * (i + 1);
        })
        .attr("y", 0)
        .style("text-anchor", "start")
        .attr("transform", function(d, i) {
            cx = size * i;
            cy = 0;
            return "rotate(-45, " + cx + ", " + cy + ")"
        })
        .attr("class", function(d, i) {
            return "context-axis";
        });
    var questionLabels = svg.append("g").selectAll(".question-label")
        .data(question_l)
        .enter().append("text")
        .text(function(d) {
            return d;
        })
        .attr("x", 0)
        .attr("y", function(d, i) {
            return size * i;
        })
        .style("text-anchor", "end")
        .attr("class", function(d, i) {
            return "question-axis";
        });


    var colorScale = d3.scaleSequential(d3.interpolateYlGnBu)
        .domain([1, 0]);

    var cards = svg.append("g")
        .attr("transform", "translate(" + font_size / 2 + ",-" + font_size + ")")
        .selectAll(".data" + i)
        .data(data_list, function(d) {
            return d.x + "-" + d.y;
        })

    var check = function(d) {
        return (d.x >= start_x && d.x <= end_x && d.y >= start_y && d.y <= end_y);
    }

    cards.enter().append("rect")
        .attr("x", function(d) {
            if (flag) {
                if (check(d)) return 1 + (d.x - start_x) * size;
                else return 0;
            } else {
                return 1 + (d.x) * size;
            }
        })
        .attr("y", function(d) {
            if (flag) {
                if (check(d)) return (d.y - start_y) * size;
                else return 0;
            } else {
                return d.y * size;
            }
        })
        .attr("class", "att-cell bordered")
        .attr("width", function(d) {
            if (!flag || check(d)) return size;
            else return 0;
        })
        .attr("height", function(d) {
            if (!flag || check(d)) return size;
            else return 0;
        })
        .attr("data-toggle", "tooltip")
        .attr("data-html", "true")
        .attr("title", function(d, i) {
            var context_w = "";
            var question_w = "";
            if (flag) {
                context_w = context_l[d.x - start_x];
                question_w = question_l[d.y - start_y];
            } else {
                context_w = context_l[d.x];
                question_w = question_l[d.y];
            }
            var word_pair = context_w + " - " + question_w;
            return word_pair + "<br/>Value: <b>" + d.v.toFixed(3) + "</b>";
        })
        .style("fill", function(d) {
            return colorScale(d.v);
        });


    cards.exit().remove();
    $('[data-toggle="tooltip"]').tooltip();
    $(".att-function").show();
    $("#tool-att-loading").text("Done! (" + data.length + ", " + +data[0].length + ")");

    $(".att-cell").click(function(e) {
        att_selected_info.push($(this));
        if (att_selected_info.length >= 2) {
            draw_att_heatmap(current_att_data);
        }
        $(this).addClass("active");
        e.stopPropagation();
        return false;
    });

    $(".att-cell").hover(function() {
        var data = $(this)[0].__data__;
        x_loc = flag ? data.x - start_x : data.x;
        y_loc = flag ? data.y - start_y : data.y;

        $(".context-axis").eq(x_loc).addClass("active");
        $(".question-axis").eq(y_loc).addClass("active");
    }, function() {
        var data = $(this)[0].__data__;
        x_loc = flag ? data.x - start_x : data.x;
        y_loc = flag ? data.y - start_y : data.y;

        $(".context-axis").eq(x_loc).removeClass("active");
        $(".question-axis").eq(y_loc).removeClass("active");
    });

    $("#tool-att-weight-svg").click(function(e) {
        att_selected_info = [];
        $(".att-cell").removeClass("active");
    });

    var legend_area = svg.append('defs')
        .append('svg:linearGradient')
        .attr('id', 'gradient')
        .attr('x1', '0%')
        .attr('y1', '100%')
        .attr('x2', '100%')
        .attr('y2', '100%')
        .attr('spreadMethod', 'pad');

    legend_area.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", "#0C2C84")
        .attr("stop-opacity", 1);

    legend_area.append("stop")
        .attr("offset", "50%")
        .attr("stop-color", "#7FCDBB")
        .attr("stop-opacity", 1);

    legend_area.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", "#FFFFD9")
        .attr("stop-opacity", 1);

    svg.append("rect")
        .attr("width", legendWidth + 2)
        .attr("height", 20)
        .attr("x", size / 2 - 3)
        .attr("y", height - 65)
        .style("fill", "url(#gradient)")

    var y = d3.scaleLinear()
        .range([0, legendWidth / 2, legendWidth])
        .domain([0, 0.5, 1]);

    var yAxis = d3.axisBottom()
        .scale(y)
        .ticks(5);

    svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + (size / 2 - 3) + "," + (height - 45) + ")")
        .call(yAxis)
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", height)
        .attr("dy", ".71em")
        .style("text-anchor", "end");

    legend_area.exit().remove();
}


function get_max(arr) {
    return Math.max.apply(null, arr);
}

function load_output_info() {
    query_text = {};
    if (view_state[0] == 0) $("#view-head-btn-ppr").click();
    $(".tool-mo-item-list").text("loading...");
    $(".att-function").hide();
    query_text["c"] = $("#view-context").text();
    query_text["q"] = $("#s_question").text();
    if (query_text["c"].length > 0 && query_text["q"].length > 0) {
        query = JSON.stringify(query_text);
        $.ajax("/model", {
            data: query,
            contentType: 'application/json',
            type: 'POST',
            success: function(data) {
                $(".tool-mo-item-list").empty();
                context_word_list = current_qa_data["context_p"].split(" ");
                context_word_list.push("&lt;noans&gt;");
                noans_index = context_word_list.length - 1;
                start_prob = $.extend([], data["start_prob"]);
                end_prob = $.extend([], data["end_prob"]);

                start_prob_max = get_max(start_prob);
                end_prob_max = get_max(end_prob);

                start_rel_prob = start_prob.map(function(x) {
                    return x / start_prob_max;
                });
                end_rel_prob = end_prob.map(function(x) {
                    return x / end_prob_max;
                });

                start_sorted_arg = argsort(start_rel_prob, true).slice(0, 30);
                end_sorted_arg = argsort(end_rel_prob, true).slice(0, 30);

                for (var i = 0; i < start_sorted_arg.length; i++) {
                    index = start_sorted_arg[i];
                    prob_value = start_prob[index];
                    prob_percent = (prob_value * 100).toFixed(2);
                    prob_color = index < (start_prob.length - 1) ? "#8effd4" : "#ff8e99";
                    back_style = prob_value < 1e-2 ? "" : "style=\"background-image: linear-gradient(to right, " +
                        prob_color + ", " + prob_color + " " + Math.floor(prob_percent) + "%, " +
                        "white " + Math.min(100, Math.floor(prob_percent) + 2) + "%);\"";

                    li_text = "<li id=\"mo-word-" + index +
                        "\" class=\"tool-mo-word\"><div class=\"tool-mo-name\" " + back_style + ">" +
                        context_word_list[index] +
                        " (" + index + ")" + "</div>" +
                        "<div class=\"tool-mo-prob\">" + prob_value.toFixed(6) + "</div></li>";
                    li_item = $.parseHTML(li_text);
                    $("#tool-mo-sp-list").append(li_item);
                }

                for (var i = 0; i < end_sorted_arg.length; i++) {
                    index = end_sorted_arg[i];
                    prob_value = end_prob[index];
                    prob_percent = (prob_value * 100).toFixed(2);
                    prob_color = index < (end_prob.length - 1) ? "#8effd4" : "#ff8e99";
                    back_style = prob_value < 1e-2 ? "" : "style=\"background-image: linear-gradient(to right, " +
                        prob_color + ", " + prob_color + " " + Math.floor(prob_percent) + "%, " +
                        "white " + Math.min(100, Math.floor(prob_percent) + 2) + "%);\"";

                    li_text = "<li id=\"mo-word-" + index +
                        "\" class=\"tool-mo-word\"><div class=\"tool-mo-name\"" + back_style + ">" +
                        context_word_list[index] +
                        " (" + index + ")" + "</div>" +
                        "<div class=\"tool-mo-prob\">" + prob_value.toFixed(6) + "</div></li>";
                    li_item = $.parseHTML(li_text);
                    $("#tool-mo-ep-list").append(li_item);
                }

                span_candidate = [];
                span_index = [];
                span_prob = [];

                span_candidate.push("<i>Unanswerable</i>");
                span_prob.push(start_prob.slice(-1)[0] * end_prob.slice(-1)[0]);
                span_index.push([-1, -1]);

                for (var i = 0; i < start_sorted_arg.length; i++) {
                    for (var j = 0; j < end_sorted_arg.length; j++) {
                        s_index = start_sorted_arg[i];
                        e_index = end_sorted_arg[j];
                        if (s_index == noans_index || e_index == noans_index) continue;
                        if (s_index > e_index) continue;
						if (Math.abs(s_index - e_index) > 15) continue;
                        s_value = start_prob[s_index];
                        e_value = end_prob[e_index];
                        span_value = s_value * e_value;
                        if (span_value < 1e-3) continue;
                        span_candidate.push(context_word_list.slice(s_index, e_index + 1).join(" "));
                        span_index.push([s_index, e_index]);
                        span_prob.push(span_value);
                    }
                }

                span_sorted_arg = argsort(span_prob, true).slice(0, 30);

                for (var i = 0; i < span_sorted_arg.length; i++) {
                    index = span_sorted_arg[i];
                    prob_value = span_prob[index];
                    prob_index = span_index[index];
                    prob_percent = (prob_value * 100).toFixed(2);
                    prob_color = index > 0 ? "#8effd4" : "#ff8e99";
                    back_style = prob_value < 1e-2 ? "" : "style=\"background-image: linear-gradient(to right, " +
                        prob_color + ", " + prob_color + " " + Math.floor(prob_percent) + "%, " +
                        "white " + Math.min(100, Math.floor(prob_percent) + 2) + "%);\"";
                    li_text = "<li id=\"mo-span-" + prob_index[0] + "_" + prob_index[1] +
                        "\" class=\"tool-mo-span\">" +
                        "<div class=\"tool-mo-name\" " + back_style + ">" + span_candidate[index] + "</div>" +
                        "<div class=\"tool-mo-prob\">" + prob_value.toFixed(6) + "</div></li>";
                    li_item = $.parseHTML(li_text);
                    $("#tool-mo-span-list").append(li_item);
                }

                var view_word = $(".view-word");
                $(".tool-mo-word").hover(function() {
                    var index = parseInt($(this).attr("id").replace("mo-word-", ""));
                    if (index < context_word_list.length - 1 && view_state[0] == 1) {
                        $($(".view-word")[index]).addClass("output_word");
                    }
                }, function() {
                    $(".view-word").removeClass("output_word");
                });

                $(".tool-mo-span").hover(function() {
                    var index = $(this).attr("id").replace("mo-span-", "").split("_");
                    var start_idx = parseInt(index[0]);
                    var end_idx = parseInt(index[1]);
                    if (start_idx < 0) return;
                    for (var i = start_idx; i < end_idx + 1; i++) {
                        $($(".view-word")[i]).addClass("output_word");
                    }
                }, function() {
                    $(".view-word").removeClass("output_word");
                });
            }
        });
    }
}

function load_sent_info() {
    $("#tool-sq-title-msg").text("similar questions with [" + current_qa_data["question"] + "]");
    $("#tool-sq-list, #tool-sq-info").empty();

    if (current_qa_data.hasOwnProperty("question")) {
        var query = {}
        query["question"] = current_qa_data["question"];
        query["context"] = current_qa_data["context"];
        query["answer"] = current_qa_data["answers"][0]["text"];
        query["num"] = 25;
        query = JSON.stringify(query);
        $.ajax("/sent", {
            data: query,
            contentType: 'application/json',
            type: 'POST',
            success: function(data) {
                sent_stat = data["stat"];
                sent_list = data["data"];
                $("#tool-sq-info").text("Total " + sent_list.length + " | " +
                    "EM " + (sent_stat[0] * 100).toFixed(2) + " | " +
                    "F1 " + (sent_stat[1] * 100).toFixed(2) + " | " +
                    "NoAns ACC " + (sent_stat[2]).toFixed(2)
                );
				var question_list = [];
                for (var i = 0; i < sent_list.length; i++) {
                    item = sent_list[i];
                    qid = item["id"];
					qname = item["name"];
					question_list.push(qid);
                    question = item["question"];
                    pred_answer = item["pred_answer"].length > 0 ? item["pred_answer"] : "<i>unanswerable</i>";
                    answer = item["answer"].length > 0 ? "(" + item["answer"] + ")" : "<i>(unanswerable)</i>";
                    label = item["label"];
                    label_class = label == 1 ? "pos" : "neg";
                    li_text = "<li class=\"" + label_class + "\"><div class=\"sq-label\"></div>" +
                        "<div id=\"sq-qa-" + qid + "\" class=\"sq-qa-item\">" +
                        "<div class=\"sq-qa-item-question\">" + question + "</div>" +
                        "<div><div class=\"sq-qa-item-col text-monospace\">" + qid + "</div>" +
						"<div class=\"sq-qa-item-col\">" + qname + "</div>" +
                        "<div class=\"sq-qa-item-answer\">" + pred_answer + " " + answer + "</div></div>" +
                        "</div></li>";

                    li_item = $.parseHTML(li_text);
                    $("#tool-sq-list").append(li_item);
                }


                $(".sq-qa-item").click(function() {
                    var qid = $(this).attr("id").replace("sq-qa-", "");
                    set_view(qid, false);
                });
            }
        });
    }
}

$("#tool-sq-tosidebar").click(function() {	
	$("#sidebar-status-text").html("loading...");
	var query_text = {};
	query_text["ids"] = $(".sq-qa-item").map(function(){return this.id.replace("sq-qa-", "")}).get();;
	var query = JSON.stringify(query_text);
	$.ajax("/ids_list", {
		data: query,
		contentType: 'application/json',
		type: 'POST',
		success: function(data) {
			load_sidebar(data);
		}
	});
});


$("#tool-adv-run").click(adv_rule_apply);

function adv_rule_apply() {
	$("#tool-adv-info").text("Loading result...")
	var query = {};
	query["id"] = current_qa;
	query["question"] = $("#s_question").text();
	query["context"] = $("#view-context").text();
	query["rules"] = $("#tool-adv-customrules").val();
	query["customrule"] = parseInt($(":radio[name=advoption]:checked").val()) == 1;
	query = JSON.stringify(query);
	$("#tool-adv-list").empty();
	$.ajax("/applyrule", {
		data: query,
		contentType: 'application/json',
		type: 'POST',
		success: function(result) {
			adv_run_flag = true;
			var orig = result["original_result"];
			var orig_em_f1 = orig["em_f1"];
			var data = result["rule_result"];
			var meta = result["meta"];
			$("#tool-adv-info").text("Matched rule(s): " + meta["matched"] + " from " + meta["total"])
			$("#tool-adv-list").empty();
			for (var i = 0; i < data.length; i++) {
				var rule = data[i]["rule"];
				var answer = data[i]["answer"];
				var na_prob = data[i]["na_prob"];
				var p_question = data[i]["question"];
				
				if (na_prob >= predict_threshold) {
					answer = "unanswerable (" + answer + ")";
				}
				
				var data_em_f1 = data[i]["em_f1"];
				var label_class = "";
				if (data_em_f1[1] > orig_em_f1[1]) label_class="pos-ex";
				if (data_em_f1[1] < orig_em_f1[1]) label_class="neg-ex";
				
				var data_em_text = (data_em_f1[0] == 1) ? "TRUE" : "FALSE";
				t = JSON.stringify(data[i]);
				li_text = "<li class=\"adv-rule-item " + label_class + "\">" 
				          + "<div class=\"col-4\" " 
						  + "data-toggle=\"tooltip\" data-placement=\"bottom\" "
						  + "title=\"" + p_question +"\""
						  + "><code>" + rule + "</code></div>"
						  + "<div class=\"col-4\">" + answer + "</div>"
						  + "<div class=\"col-2\"><span>" 
						  + data_em_text + " / " 
						  + (data_em_f1[1]*100).toFixed(2) + "</span>"
						  + "</div>"
						  + "<div class=\"col-2\">" + na_prob.toFixed(3) + "</div>"
						  + "</li>";
				li_item = $.parseHTML(li_text);
				$("#tool-adv-list").append(li_item);				
			}
			
			$('[data-toggle="tooltip"]').tooltip();
		}
	});
}

$(document).keydown(function(event) {
    keydown_state = event.which;
});

$(document).keyup(function(event) {
    keydown_state = 0;
});

$("#tool-menu .nav-item").click(function() {
    var index = $(this).index();
    set_menu(index);
    if (index != 0) $(".view-word").removeClass("selected-word");
    if (index == 4) {
        $("#view-context").addClass("noselect");
        $(".view-qa-text").addClass("noselect");
    } else {
        $("#view-context").removeClass("noselect");
        $(".view-qa-text").removeClass("noselect");
    }
    if (index == 1) load_att_info();
    if (index == 2) load_output_info();
    if (index == 3) load_sent_info();
});

$(".view-head-btn").click(function() {
    var index = $(this).index();
    if (view_state[index] > 0) {
        view_state[index] = 0;
        $(this).removeClass("active");
    } else {
        view_state[index] = 1;
        $(this).addClass("active");
    }
    if (current_qa.length > 0) set_view(current_qa, false);
});

$("#view-context").scroll(function(e) {
    var newScroll = $(this).scrollTop();
    $("#view-context-back").scrollTop(newScroll);
});

$('input[type=radio][name=advoption]').change(function() {
    if (this.value == '0') {
        $("textarea[name=advcustomrule]").prop('disabled', true);
    } else {
        $("textarea[name=advcustomrule]").prop('disabled', false);
    }
});

$("#modal-edit-apply-btn").click(function() {
    var src = $("#modal-edit-src").val();
    var index = parseInt($("#modal-edit-idx").val());
    var old_text = $("#" + src + " .view-word").eq(index).text();
    var new_text = $("#modal-edit-new").val();
	var new_text_size = new_text.split(" ").length;
	console.log("change: " + old_text + " -> " + new_text);

    var new_element = $("#" + src).clone();
	new_element.find(".view-word").eq(index).text(new_text);
    var context = $("#view-context").text();
	var question = $("#s_question").text();

	var pred_answer = $("s_answer_pred").text();

	if (old_text != new_text) {
        $("#view-head").removeClass("pos-ex").removeClass("neg-ex");
    }
	
	if (src == "view-context") {
		context = new_element.text();
	} else {
		question = new_element.text();
	}

	var query_text = {};
    query_text["id"] = current_qa;
	query_text["context"] = context;
	query_text["question"] = question;
	
	query = JSON.stringify(query_text);
	$.ajax("/eval", {
		data: query,
		contentType: 'application/json',
		type: 'POST',
		success: function(result) {
			current_qa = result["id"];
			set_view_html(result, false);
			for (var i = 0; i < new_text_size; i++) {
				$("#" + src + " .view-word").eq(index + i).addClass("fixed_word");
			}
			$('#modal-edit').modal('hide');
		}
	});
});


$(document).ready(function() {
	if (localStorage.getItem("filter") === null) {
		filter["label"] = [1, 1];
		filter["pred"] = [1, 1];
		filter["prob_expr"] = "";
		filter["limit"] = 500;
		filter["shuffle"] = true;
		
		localStorage.setItem("filter", JSON.stringify(filter));
	} else {
		filter = JSON.parse(localStorage.filter);
	}

	if (localStorage.getItem("bookmark") === null) {
		localStorage.setItem("bookmark", JSON.stringify(bookmark));
	} else {
		bookmark = JSON.parse(localStorage.bookmark);
	}
	
    $('[data-toggle="tooltip"]').tooltip();
    $(".view-head-btn").each(function() {
        view_state[$(this).index()] = $(this).hasClass("active");
    });
	init_filter_modal();
    fetch_ids();
    $("#tool-em-title").html("select word in context or question (alt key: compare two words)");
    $("#tool-em-info").hide();
});

$("#tool-att-reset").click(function() {
    att_selected_info = [];
    load_att_info();
});

$("#tool-att-save").click(function() {
    if (current_qa.length == 0) return;
    var filename = "att_c2q_" + current_qa;
    var svg = document.getElementById("tool-att-weight-svg");
    var serializer = new XMLSerializer();
    var svg_data = serializer.serializeToString(svg);
    var svg_blob = new Blob([serializer.serializeToString(svg)], {
        'type': "image/svg+xml"
    });
    var url = URL.createObjectURL(svg_blob);

    var canvas = document.createElement("canvas");
    var svgSize = svg.getBoundingClientRect();
    canvas.width = svgSize.width;
    canvas.height = svgSize.height;

    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    var img = document.createElement("img");
    img.setAttribute("src", "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svg_data))));

    img.onload = function() {
        ctx.drawImage(img, 0, 0);
        var imgsrc = canvas.toDataURL("image/png");
        var a = document.createElement("a");
        a.download = filename + ".png";
        a.href = imgsrc;
        a.click();
    };
});

$("#expand-up").click(function() {
    expand_state = Math.max(-1, expand_state - 1);
    make_expand();
});

$("#expand-down").click(function() {
    expand_state = Math.min(1, expand_state + 1);
    make_expand();
});

$('#sidebar-search').on('input', function(e) {
    clearTimeout(search_timeout);
    search_timeout = setTimeout(retrieve_search_result, 500);
});

$("#sidebar-refresh-btn").click(fetch_ids);

$("#view-head-btn-eval.active").click(function() {
	
});

function retrieve_search_result() {
	var query_text = {}
	query_text["query"] = $('#sidebar-search input').val();
	query_text["filter"] = filter;	
	query = JSON.stringify(query_text);
	$.ajax("/search", {
		data: query,
		contentType: 'application/json',
		type: 'POST',
		success: function(data) {
			$("#sidebar-id-list").empty();
			$("#sidebar-status-text").html("Total: " + data["data"].length + " (" + data["ratio"] + "%)");
			for (var i in data["data"]) {
				item = data["data"][i];
				ua_tag = item["uans"] == (item["pred"] > predict_threshold) ? "pos-ex" : "neg-ex";
				$("#sidebar-id-list").append(
					"<li id=\"qa-" + item["key"] + "\" class=\"qa-item " + ua_tag + "\">" +
					"<div>" + 
					"<div class=\"sidebar-qid text-monospace\">" + item["key"] + "</div>" +
					"<div class=\"sidebar-name\">" + item["name"] + "</div>" +
					"<div class=\"sidebar-question\">" + item["q"] + "</div>" +
					"</div></li>");
				$("#qa-" + item["key"]).click(function() {
					set_view($(this).attr("id").replace("qa-", ""), true);
				});
			}
		}
	});
}

function make_expand() {
    if (expand_state == 0) {
        $(".expanded").removeClass("expanded");
        $(".hidden").removeClass("hidden");
    } else if (expand_state > 0) {
        $("#tool-wrapper").addClass("hidden");
        $("#view-wrapper").addClass("expanded");
    } else if (expand_state < 0) {
        $("#tool-wrapper").addClass("expanded");
        $("#view-wrapper").addClass("hidden");
    }
    $("#expand-bar").attr("state", expand_state);
}

$("#sidebar-bookmark-btn").click(function() {	
	$("#sidebar-status-text").html("loading...");
	var query_text = {};
	query_text["ids"] = bookmark;
	var query = JSON.stringify(query_text);
	$.ajax("/ids_list", {
		data: query,
		contentType: 'application/json',
		type: 'POST',
		success: function(data) {
			load_sidebar(data);
		}
	});
});

function show_bookmark(qid) {
	var btn = $("#bookmark_stat");
	btn.removeClass("bm-on");
	
	if (bookmark.includes(qid)) {
		btn.addClass("bm-on");
		btn.attr("src", "files/head_b_on.png");
	} else {
		btn.removeClass("bm-on");
		btn.attr("src", "files/head_b_off.png");
	}	
}

function change_bookmark(qid) {
	var btn = $("#bookmark_stat");
	var bookmark_stat = btn.hasClass("bm-on");
	
	if (bookmark_stat) {
		btn.removeClass("bm-on");
		btn.attr("src", "files/head_b_off.png");
		var element_idx = bookmark.indexOf(qid);
		bookmark.splice(element_idx, 1);
	} else {
		btn.addClass("bm-on");
		btn.attr("src", "files/head_b_on.png");
		bookmark.push(qid);
	}
	localStorage.setItem("bookmark", JSON.stringify(bookmark));
}
