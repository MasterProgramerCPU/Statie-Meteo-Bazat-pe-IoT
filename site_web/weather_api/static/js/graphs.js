document.addEventListener('DOMContentLoaded', function() {
    Promise.all([
        fetch('/api/temperatura/').then(response => response.json()),
        fetch('/api/presiune/').then(response => response.json()),
        fetch('/api/umiditate/').then(response => response.json()),
        fetch('/api/lumina/').then(response => response.json()),
        fetch('/api/ceata/').then(response => response.json()),
        fetch('/api/anemometru/').then(response => response.json())
    ]).then(dataArrays => {
        createLineChart(dataArrays[0], 'temperature-chart', 'valoarea_medie', 'red', 'Temperatura');
        createLineChart(dataArrays[1], 'pressure-chart', 'valoare', 'white', 'Presiune');
        createLineChart(dataArrays[2], 'humidity-chart', 'valoarea_medie', 'turquoise', 'Umiditate');
        createLineChart(dataArrays[3], 'light-chart', 'valoarea_medie', 'yellow', 'Lumina');
        createLineChart(dataArrays[4], 'fog-chart', 'ceata', 'gray', 'Ceață');
        createLineChart(dataArrays[5], 'wind-chart', 'valoarea_medie', 'blue', 'Vânt');
    }).catch(error => console.error('Error fetching data:', error));
});

function createLineChart(data, elementId, valueField, color, label) {
    // Clear the existing graph container
    d3.select(`#${elementId}`).selectAll("*").remove();

    const margin = { top: 40, right: 30, bottom: 80, left: 60 },
          width = 460 - margin.left - margin.right,
          height = 400 - margin.top - margin.bottom;

    const svg = d3.select(`#${elementId}`)
                  .append("svg")
                  .attr("width", width + margin.left + margin.right)
                  .attr("height", height + margin.top + margin.bottom)
                  .append("g")
                  .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear()
                .domain(d3.extent(data, d => d.id))
                .range([ 0, width ]);

    const y = d3.scaleLinear()
                .domain([d3.min(data, d => d[valueField]), d3.max(data, d => d[valueField])])
                .range([ height, 0 ]);

    const line = svg.append("path")
                    .datum(data)
                    .attr("fill", "none")
                    .attr("stroke", color)
                    .attr("stroke-width", 2.5)
                    .attr("d", d3.line()
                                 .x(d => x(d.id))
                                 .y(d => y(d[valueField])));

    const xAxis = svg.append("g")
                     .attr("transform", `translate(0,${height})`)
                     .attr("class", "axis")
                     .call(d3.axisBottom(x));

    const yAxis = svg.append("g")
                     .attr("class", "axis")
                     .call(d3.axisLeft(y));

    svg.append("text")
       .attr("x", width / 2)
       .attr("y", -20)
       .attr("text-anchor", "middle")
       .attr("font-size", "16px")
       .attr("fill", color)
       .text(label);

    const zoom = d3.zoom()
                   .scaleExtent([1, 10])
                   .translateExtent([[-width, -height], [width * 2, height * 2]])
                   .on("zoom", zoomed);

    svg.call(zoom);

    function zoomed(event) {
        const transform = event.transform;
        const newX = transform.rescaleX(x);
        const newY = transform.rescaleY(y);
        xAxis.call(d3.axisBottom(newX));
        yAxis.call(d3.axisLeft(newY));
        line.attr("d", d3.line()
                          .x(d => newX(d.id))
                          .y(d => newY(d[valueField])));
    }

    const tooltip = d3.select("body").append("div")
                      .attr("class", "tooltip")
                      .style("opacity", 0);

    svg.selectAll("circle")
       .data(data)
       .enter().append("circle")
       .attr("r", 0) // Set radius to 0 to remove the circles
       .attr("cx", d => x(d.id))
       .attr("cy", d => y(d[valueField]))
       .attr("fill", color)
       .attr("stroke", "white")
       .on("mouseover", function(event, d) {
           tooltip.transition()
                  .duration(200)
                  .style("opacity", .9);
           tooltip.html(`${label}: ${d[valueField]}<br/>ID: ${d.id}`)
                  .style("left", (event.pageX + 5) + "px")
                  .style("top", (event.pageY - 28) + "px");
       })
       .on("mouseout", function(d) {
           tooltip.transition()
                  .duration(500)
                  .style("opacity", 0);
       });

    // Calculate mean and dispersion
    const mean = d3.mean(data, d => d[valueField]);
    const dispersion = d3.deviation(data, d => d[valueField]);

    // Add mean and dispersion text
    svg.append("text")
       .attr("x", width / 2)
       .attr("y", height + 40)
       .attr("text-anchor", "middle")
       .attr("font-size", "14px")
       .attr("fill", color)
       .text(`Media = ${mean.toFixed(2)}`);

    svg.append("text")
       .attr("x", width / 2)
       .attr("y", height + 60)
       .attr("text-anchor", "middle")
       .attr("font-size", "14px")
       .attr("fill", color)
       .text(`Dispersia = ${dispersion.toFixed(2)}`);
}
