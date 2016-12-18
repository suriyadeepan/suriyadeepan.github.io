var grid = [];
var grid_steps = 50;
var t = 0.0;
var t_1 = -1.0; // memory
// assign rule set here
var ruleset_dec = 30;

var cell_scale = 12;

// decimal to array of bits in boolean
function dec2array(dec_num){
	
	// array of bits in boolean
	var arr = [];

	// to string
	var base2 = (dec_num).toString(2);

	// iterate string
	for(var j=0;j<base2.length;j++){
		arr.push(boolean(int(base2[j])));
	}

	arr_len = arr.length;
	// prepend zeros
	for(var j=0;j<8-arr_len;j++){
		arr.unshift(false);
	}

	return arr;
}

function array2dec(arr){
	var dec = 0;

	for(var i=arr.length-1;i>=0;i--){
		dec += int(arr[i]) * pow(2,arr.length-i-1);
	}

	return dec;
}

function array2string(arr){
	var arr_str = '';
	for(var i=0;i<arr.length;i++){
		arr_str += str(int(arr[i]));
	}
	return arr_str;
}

function setup(){
	
	//cell_scale = 10;

	createCanvas(70*cell_scale, 55*cell_scale);
	background(50);

	//renderRule(70);
}

function renderRule(rulenum){

	ruleset = dec2array(rulenum);
	background(50);
	translate(50,10);

	// the parent row
	grid = new Grid(init_cells(grid_steps), 1, cell_scale);
	grid.render_text();

	// create 100 generations
	for(var i=0;i<grid_steps;i++){
		// get next gen
		grid = grid.next_gen(ruleset);
		// display
		//grid.show();
		grid.render_text();
	}


}

function random_cells(len){
	var cells = [];
	for(var i=0;i<len;i++){
		cells.push(Math.random() >= 0.5);
	}
	return cells;
}

function init_cells(len){
	var cells = new Array(len).fill(false);
	cells[len/2] = true;
	return cells;
}

function draw(){
	
	
	// reset time
	if(t>255){
		t=0;
	}

	if(int(t_1) != int(t)){
		renderRule(t);
		t_1 = t;
	}

	/*
	// ruleset - incremented every frame (not exactly)
	ruleset = dec2array(int(t));
	background(50);
	translate(50,20);

	// the parent row
	grid = new Grid(init_cells(grid_steps),1);
	grid.show();

	// create 100 generations
	for(var i=0;i<100;i++){
		// get next gen
		grid = grid.next_gen(ruleset);
		// display
		grid.show();
	}

	// add text to show ruleset number
	textSize(64);
	push();
	fill(0, 102, 153);
	stroke(0,102,153);
	text(str(int(t)), 200, 575);
	pop();
	*/
	// time increment
	t+=0.01;
	
}

