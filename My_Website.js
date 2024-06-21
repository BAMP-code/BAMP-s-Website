const aboutMe = document.querySelector('.aboutme-page');
const initialTop = parseInt(window.getComputedStyle(aboutMe).top);

//listen for the scroll
window.addEventListener('scroll', () => {
    const scrollPosition = window.scrollY;
    const scrollStart = 50;
    if (scrollPosition > scrollStart) {
        aboutMe.style.transition = 'top 1.5s ease'
        aboutMe.style.top = '230px'
    } else {
        aboutMe.style.transition = 'top 1.5s ease'
        aboutMe.style.top = initialTop + 'px' }
});


function showSidebar() {
    const sidebar = document.querySelector('.side-bar')
    sidebar.style.display = 'flex'
}

function hideSidebar() {
    const sidebar = document.querySelector('.side-bar')
    sidebar.style.display = 'none'
}

//Project Sliders
let prev = document.querySelector('.prev');
let next = document.querySelector('.next');
let slider = document.querySelector('.slider');

next.addEventListener('click', function() {
    let slides = document.querySelectorAll('.slides');
    slider.appendChild(slides[0]);
})
prev.addEventListener('click', function() {
    let slides = document.querySelectorAll('.slides');
    slider.prepend(slides[slides.length - 1]);
})

document.addEventListener('DOMContentLoaded', function () {
    const sidebarLinks = document.querySelectorAll('.side-bar a');
    const sidebar = document.querySelector('.side-bar');

    sidebarLinks.forEach(link => {
        link.addEventListener('click', function (event) {

            if (this.getAttribute('href').startsWith('mailto:')) {
                return;
            }
            event.preventDefault();
            
            hideSidebar();

            const targetId = this.getAttribute('href');

            document.querySelector(targetId).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const content = document.querySelectorAll('.slider .slides:nth-child(2) .content');

    content.forEach(cont => {
        cont.addEventListener('touchstart', function () {
            this.classList.add('touch-active');
        });
        cont.addEventListener('touchend', function () {
            setTimeout(() => {
                this.classList.remove('touch-active');
            }, 2000);
        });
    });
});


