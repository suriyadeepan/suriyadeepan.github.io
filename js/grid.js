function Grid(cells, generation, cell_scale){

	this.text = " all work and no play makes jack a dull boy ";
	// cells
	this.cells = cells;
	this.generation = generation;
	// scale
	this.cell_scale = cell_scale;
	// create next generation
	this.next_gen = function(ruleset){
		//return new Grid(this.cells, this.generation + 1);

		// next gen cells
		var n_cells = [];
		n_cells.push(cells[0]);

		// iterate through cells
		for(var i=1;i<cells.length -1;i++){
			// convert bit arrays to decimal
			neighborhood = cells[i-1] << 2 | cells[i] << 1 | cells[i+1];
			// fill in next gen
			n_cells.push(ruleset[ruleset.length - neighborhood -1]);
		}
		n_cells.push(cells[cells.length -1]);

		return new Grid(n_cells, this.generation +1, this.cell_scale);
	}

	

	this.show = function(){
		stroke(190);
		for(var i=0;i<cells.length;i++){
			if(cells[i]){
				rect(5*i,5*generation,5,5)
			}
		}
		

	}

	this.render_text = function(){
		push();
		stroke(190);
		fill(190);
		textSize(this.cell_scale);
		textFont("Courier New");
		var text_idx = 0;
		for(var i=0;i<cells.length;i++){
			if(cells[i]){
				//rect(5*i,5*generation,5,5)
				
				if(text_idx >= this.text.length){
					text_idx = 0;
				}
				else{
					text(str(this.text[text_idx]), this.cell_scale*i, this.cell_scale*generation);
					text_idx++;
				}
			}
		}
		pop();		
	}
}