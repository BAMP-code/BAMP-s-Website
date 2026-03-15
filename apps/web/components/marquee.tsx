export function Marquee() {
  return (
    <div className="w-[140%] overflow-hidden px-[1280px] py-[90px]">
      <h3 className="section-title-gradient text-[90px] font-[500] tracking-[0.34em] opacity-70 lg:text-[60px]">
        <div className="w-[100000px] animate-marquee">
          <div className="mr-5 float-left">BAMP</div>
        </div>
      </h3>
    </div>
  );
}
