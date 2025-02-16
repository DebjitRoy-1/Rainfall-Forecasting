document.addEventListener('DOMContentLoaded', () => {
    // Initialize Chart
    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.error('Canvas Element not found');
        return;
    }

    const ctx = chartElement.getContext('2d'); // Get context to draw on canvas

    // Cool Neon Gradient Color for the Line Chart (Adding bright colors for glowing effect)
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(0, 204, 255, 1)'); // Bright cyan
    gradient.addColorStop(1, 'rgba(255, 44, 255, 1)'); // Vibrant purple

    // Example forecast items (Replace with dynamic data)
    const forecastItems = [
        { time: '12:00 PM', temperature: 22, humidity: 65 },
        { time: '1:00 PM', temperature: 24, humidity: 60 },
        { time: '2:00 PM', temperature: 26, humidity: 58 },
        { time: '3:00 PM', temperature: 28, humidity: 55 },
        { time: '4:00 PM', temperature: 30, humidity: 53 }
    ];

    const times = [];
    const temps = [];

    // Collect data from forecast items (time, temperature)
    forecastItems.forEach(item => {
        const time = item.time; // Time text
        const temp = item.temperature; // Temperature text

        if (time && temp !== undefined) {
            times.push(time);
            temps.push(temp);
        }
    });

    // Validate collected data
    if (temps.length === 0 || times.length === 0) {
        console.error('Temperature or time values are missing.');
        return;
    }

    // Create the Chart with Enhanced Styling
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: times, // x-axis labels (time)
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: temps, // y-axis data (temperature values)
                    borderColor: gradient,
                    borderWidth: 4,
                    tension: 0.4,
                    pointRadius: 6,
                    pointBackgroundColor: gradient,
                    pointBorderColor: '#fff',
                    pointHoverRadius: 12,
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: gradient,
                    fill: true, // Fill area under the line
                    backgroundColor: gradient,
                    hoverBorderWidth: 3,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true, // Show legend
                    labels: {
                        font: {
                            size: 16,
                            color: '#fff', // White font color for the legend
                        },
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)', // Dark tooltip background
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    bodyFont: {
                        size: 14,
                    },
                    titleFont: {
                        size: 16,
                        weight: 'bold',
                    },
                    padding: 10,
                    callbacks: {
                        label: function (context) {
                            return `Temperature: ${context.raw}°C`; // Custom tooltip content
                        },
                    },
                },
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        drawOnChartArea: false,
                    },
                    ticks: {
                        font: {
                            size: 14,
                            weight: 'bold',
                        },
                        color: '#fff', // White color for x-axis ticks (for better visibility)
                    },
                },
                y: {
                    display: true,
                    grid: {
                        drawOnChartArea: true,
                        color: 'rgba(255, 255, 255, 0.2)', // Subtle white grid lines for the y-axis
                    },
                    ticks: {
                        font: {
                            size: 14,
                            weight: 'bold',
                        },
                        color: '#fff', // White color for y-axis ticks (for better visibility)
                    },
                },
            },
            animation: {
                duration: 1200, // Smooth animation for chart loading
                easing: 'easeOutBounce', // Bouncy effect for loading
            },
            elements: {
                line: {
                    tension: 0.4, // Smooth curve for the line
                },
                point: {
                    radius: 8, // Larger point size
                    hoverRadius: 12, // Larger size on hover
                    hitRadius: 15,
                },
            },
            hover: {
                mode: 'nearest',
                intersect: false,
            },
        },
    });

    console.log('Chart created successfully!');
});


