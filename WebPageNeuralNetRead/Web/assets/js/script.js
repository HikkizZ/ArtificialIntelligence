document.addEventListener("DOMContentLoaded", function () {
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 1;

    var mousedown = false;

    canvas.onmousedown = function (e) {
        var pos = fixPosition(e, canvas);
        const context = canvas.getContext('2d');

        context.clearRect(0, 0, canvas.width, canvas.height);
        mousedown = true;
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        return false;
    };

    canvas.onmousemove = function (e) {
        var pos = fixPosition(e, canvas);
        if (mousedown) {
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
        }
    };

    canvas.onmouseup = function (e) {
        mousedown = false;

        var pixels = [];
        for (var x = 0; x < 28; x++) {
            for (var y = 0; y < 28; y++) {
                var imgData = ctx.getImageData(y, x, 1, 1);
                var data = imgData.data;

                var color = (data[3]) / 255;
                color = (Math.round(color * 100) / 100).toFixed(2)
                pixels.push(color);
            }
        }

        console.log(pixels);

        $.post("http://localhost:8000", { pixeles: pixels.join(",") },
            function (response) {
                console.log("Resultado: " + response);
                $("#resultado").html(response);
            }
        );
    };

    function fixPosition(e, gCanvasElement) {
        var x;
        var y;
        if (e.pageX || e.pageY) {
            x = e.pageX;
            y = e.pageY;
        }
        else {
            x = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
            y = e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
        }
        x -= gCanvasElement.offsetLeft;
        y -= gCanvasElement.offsetTop;
        return { x: x, y: y };
    }
});
