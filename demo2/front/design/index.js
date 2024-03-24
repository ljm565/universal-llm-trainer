function darkMode(self){
    if (self.value === 'dark'){
        self.value = 'light';
        document.getElementById('modeState').innerHTML = '라이트 모드로 보기';
        $('body').css('color', 'white');
        $('body').css('background-color', 'rgb(50, 50, 50)');
        $('#mainHead h1').css('color', 'rgb(200, 200, 200)');
        $('#mainHead h1').css('background-color', 'rgb(50, 50, 50)');
        $('.highlight').css('color', 'rgb(159, 204, 255)');
        $('#instructions').css('background-color', 'rgb(26, 25, 25)');
        $('#instructions').css('color', 'white');
        hoveringOn(self);
    } else{
        self.value = 'dark'
        document.getElementById('modeState').innerHTML = '다크 모드로 보기';
        $('body').css('color', 'black');
        $('body').css('background-color', 'white');
        $('#mainHead h1').css('color', 'rgb(82, 82, 82)');
        $('#mainHead h1').css('background-color', 'white');
        $('.highlight').css('color', 'rgb(0, 3, 206)');
        $('#instructions').css('background-color', 'rgb(255, 255, 255)');
        $('#instructions').css('color', 'black');
        hoveringOn(self);
    }
}

function hoveringOn(self){
    if (self.value === 'dark'){
        document.getElementById('modeImg').src = 'design/index_img/moon_on.png';
        $(self).css('background-color', 'rgb(80, 80, 80)');
        $(self).css('color', 'white');
    } else{
        document.getElementById('modeImg').src = 'design/index_img/sun_on.png';
        $(self).css('background-color', 'rgb(224, 224, 224)');
        $(self).css('color', 'black');
    }
}

function hoveringOff(self){
    if (self.value === 'dark'){
        document.getElementById('modeImg').src = 'design/index_img/moon_off.png';
        $(self).css('background-color', 'rgb(224, 224, 224)');
        $(self).css('color', 'black');
    } else{
        document.getElementById('modeImg').src = 'design/index_img/sun_off.png';
        $(self).css('background-color', 'rgb(80, 80, 80)');
        $(self).css('color', 'white');
    }
}


function reload(){
    $('#container').css('opacity', 1);
    $('#mainHead h1').fadeIn(50); 
}


function headHighlightColorChanger(){
    $('a').css('font-weight', 'initial');
    if ($('#modeButton button').val() === 'dark'){
        $('#mainHead h1').css('color', 'rgb(82, 82, 82)');
        $('#mainHead h1').css('background-color', 'white');
        $('.highlight').css('color', 'rgb(0, 3, 206)');
    } else{
        $('#mainHead h1').css('color', 'rgb(200, 200, 200)');
        $('#mainHead h1').css('background-color', 'rgb(50, 50, 50)');
        $('.highlight').css('color', 'rgb(159, 204, 255)');
    }
}


function detectScroll(){
    var didScroll; 
    var lastScrollTop = 0;
    var delta = 5;  
    
    $(window).scroll(function(){ 
        didScroll = true;
    });

    setInterval(function() { 
        if ($(window).scrollTop() <= 0) {
            $('#mainHead h1').fadeIn(50);
            didScroll = false;
        } 
        else if ($(window).scrollTop() + $(window).innerHeight() + 10 > $('body').prop('scrollHeight')) {
            lastScrollTop = $(window).scrollTop();
            if ($(window).scrollTop() < lastScrollTop){ 
                $('#mainHead h1').fadeIn(50); 
            }
            didScroll = false;
        } 
        else if (didScroll) { 
            hasScrolled(); 
            didScroll = false; 
        }
    }, 10); 

    function hasScrolled() {
        var st = $(window).scrollTop();

        if(Math.abs(lastScrollTop - st) <= delta){
            return;
        }

        if (st > lastScrollTop){ 
            $('#mainHead h1').fadeOut(50); 
        } else{
            $('#mainHead h1').fadeIn(50); 
        }
        lastScrollTop = st;
    }
}


function connectWebSocket() {
    socket = new WebSocket('ws://localhost:8000/ws/stream');

    socket.onopen = function(event) {
        console.log('WebSocket connected');
    };

    socket.onerror = function(event) {
        console.error('WebSocket error:', event);
    };

    // 서버로부터 메시지를 받았을 때 실행할 함수
    socket.onmessage = function(event) {
        responseDiv.innerHTML += event.data;
    };

    return socket
}